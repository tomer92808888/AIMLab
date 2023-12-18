import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        ecg_features = config["ecg_features"]
        transformer_heads = config["transformer_heads"]
        transformer_ff_features = config["transformer_ff_features"]
        transformer_activation = config["transformer_activation"]
        transformer_layers = config["transformer_layers"]
        transformer_sequence_length = config["transformer_sequence_length"]
        spectrogram_encoder_channels = config["spectrogram_encoder_channels"]
        spectrogram_encoder_spans = config["spectrogram_encoder_spans"]
        latent_vector_features = config["latent_vector_features"]
        classes = config["classes"]
        activation = config["activation"]
        dropout = config["dropout"]

        # Initializing ECG encoder
        self.ecg_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=ecg_features,
                nhead=transformer_heads,
                dim_feedforward=transformer_ff_features,
                dropout=dropout,
                activation=transformer_activation),
            num_layers=transformer_layers,
            norm=nn.LayerNorm(normalized_shape=ecg_features)
        )

        # Initializing positional embedding
        self.positional_embedding = nn.Parameter(
            0.1 * torch.randn(transformer_sequence_length, 1, ecg_features),
            requires_grad=True
        )

        # Initializing spectrogram encoder
        self.spectrogram_encoder = nn.ModuleList()
        for index, (spectrogram_encoder_channel, spectrogram_encoder_span) in \
                enumerate(zip(spectrogram_encoder_channels, spectrogram_encoder_spans)):
            if spectrogram_encoder_span is None:
                self.spectrogram_encoder.append(ResidualConvBlock(
                    in_channels=spectrogram_encoder_channel[0], out_channels=spectrogram_encoder_channel[1],
                    latent_vector_features=latent_vector_features,
                    activation=activation, dropout=dropout)
                )
            else:
                self.spectrogram_encoder.append(AxialAttentionBlock(
                    in_channels=spectrogram_encoder_channel[0], out_channels=spectrogram_encoder_channel[1],
                    span=spectrogram_encoder_span, latent_vector_features=latent_vector_features,
                    activation=activation, dropout=dropout)
                )

        # Initializing final linear layers
        self.linear_layer_1 = nn.Sequential(
            nn.Linear(in_features=spectrogram_encoder_channels[-1][1],
                      out_features=latent_vector_features, bias=True),
            activation()
        )
        self.linear_layer_2 = nn.Linear(
            in_features=latent_vector_features, out_features=classes, bias=True
        )

        # Flags for ablation studies
        self.no_spectrogram_encoder: bool = False
        self.no_signal_encoder: bool = False

    def forward(self, ecg_lead, spectrogram):
        if self.no_spectrogram_encoder:
            latent_vector = self.__encode_ecg_lead(ecg_lead)
            output = self.linear_layer_2(latent_vector)
        elif self.no_signal_encoder:
            output = self.__process_spectrogram(spectrogram)
        else:
            latent_vector = self.__encode_ecg_lead(ecg_lead)
            output = self.__process_combined(spectrogram, latent_vector)

        return output.softmax(dim=-1) if not self.training else output

    def __encode_ecg_lead(self, ecg_lead):
        encoded = self.ecg_encoder(ecg_lead.permute(1, 0, 2) + self.positional_embedding)
        return encoded.permute(1, 0, 2).mean(dim=1)

    def __process_spectrogram(self, spectrogram):
        for block in self.spectrogram_encoder:
            spectrogram = block(spectrogram, None)
        spectrogram = F.adaptive_avg_pool2d(spectrogram, output_size=(1, 1))
        return self.linear_layer_2(self.linear_layer_1(spectrogram.flatten(start_dim=1)))

    def __process_combined(self, spectrogram, latent_vector):
        for block in self.spectrogram_encoder:
            spectrogram = block(spectrogram, latent_vector)
        spectrogram = F.adaptive_avg_pool2d(spectrogram, output_size=(1, 1))
        combined_output = self.linear_layer_1(spectrogram.flatten(start_dim=1))
        return self.linear_layer_2(combined_output + latent_vector)

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, latent_vector_features=256,
                 kernel_size=(3, 3), stride=(1, 1),
                 padding=(1, 1), bias=False, convolution=nn.Conv2d,
                 activation=nn.PReLU, pooling=nn.AvgPool2d,
                 dropout=0.0):
        super(ResidualConvBlock, self).__init__()

        # Main mapping layers
        self.main_mapping_conv_1 = convolution(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)
        self.main_mapping_norm_1 = ConditionalBatchNormLayer(out_channels, latent_vector_features)
        self.main_mapping_act_1 = activation()
        self.main_mapping_dropout_1 = nn.Dropout2d(p=dropout)
        self.main_mapping_conv_2 = convolution(out_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.main_mapping_norm_2 = ConditionalBatchNormLayer(out_channels, latent_vector_features)

        # Residual mapping
        self.residual_mapping = convolution(in_channels, out_channels, (1, 1), (1, 1), (0, 0), bias=False) \
                                if in_channels != out_channels else nn.Identity()

        # Final activation and dropout
        self.final_activation = activation()
        self.dropout = nn.Dropout2d(p=dropout)

        # Downsampling layer
        self.pooling = pooling((2, 2), (2, 2))

    def forward(self, input, latent_vector):
        output = self.main_mapping_conv_1(input)
        output = self.main_mapping_norm_1(output, latent_vector)
        output = self.main_mapping_act_1(output)
        output = self.main_mapping_dropout_1(output)
        output = self.main_mapping_conv_2(output)
        output = self.main_mapping_norm_2(output, latent_vector)
        output += self.residual_mapping(input) # Skip connection
        output = self.final_activation(output)
        output = self.dropout(output)
        return self.pooling(output)

class ConditionalBatchNormLayer(nn.Module):
    def __init__(self, num_features, latent_vector_features, track_running_stats=True):
        super(ConditionalBatchNormLayer, self).__init__()
        self.batch_normalization = nn.BatchNorm2d(num_features, track_running_stats=track_running_stats, affine=True)
        self.linear_mapping = nn.Linear(latent_vector_features, 2 * num_features, bias=False)

    def forward(self, input, latent_vector):
        normalized = self.batch_normalization(input)

        if latent_vector is not None:
            scale_bias = self.linear_mapping(latent_vector)
            scale, bias = scale_bias.chunk(2, dim=-1)
            scale = scale.unsqueeze(-1).unsqueeze(-1)
            bias = bias.unsqueeze(-1).unsqueeze(-1)
            normalized = normalized * scale + bias

        return normalized

class AxialAttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dim, span, groups=16):
        super(AxialAttentionLayer, self).__init__()
        assert (in_channels % groups == 0) and (out_channels % groups == 0), \
            "In and output channels must be a factor of the utilized groups."

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.span = span
        self.groups = groups
        self.group_channels = out_channels // groups

        # Initialize layers
        self.query_key_value_mapping = nn.Sequential(
            nn.Conv1d(in_channels, 2 * out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm1d(2 * out_channels, track_running_stats=True, affine=True)
        )
        self.output_normalization = nn.BatchNorm1d(2 * out_channels, track_running_stats=True, affine=True)
        self.similarity_normalization = nn.BatchNorm2d(3 * self.groups, track_running_stats=True, affine=True)
        self.relative_embeddings = nn.Parameter(torch.randn(2 * self.group_channels, 2 * self.span - 1), requires_grad=True)

        relative_indexes = torch.arange(self.span, dtype=torch.long).unsqueeze(dim=1) \
                           - torch.arange(self.span, dtype=torch.long).unsqueeze(dim=0) \
                           + self.span - 1
        self.register_buffer("relative_indexes", relative_indexes.view(-1))

    def forward(self, input):
        input = input.permute(0, 3, 1, 2) if self.dim == 0 else input.permute(0, 2, 1, 3)
        batch_size, dim_1, channels, dim_attention = input.shape
        input = input.reshape(batch_size * dim_1, channels, dim_attention).contiguous()

        query_key_value = self.query_key_value_mapping(input)
        query, key, value = query_key_value.reshape(batch_size * dim_1, self.groups, self.group_channels * 2, dim_attention).split([self.group_channels // 2, self.group_channels // 2, self.group_channels], dim=2)
        embeddings = self.relative_embeddings.index_select(dim=1, index=self.relative_indexes).view(2 * self.group_channels, self.span, self.span)
        query_embedding, key_embedding, value_embedding = embeddings.split([self.group_channels // 2, self.group_channels // 2, self.group_channels], dim=0)

        query_embedded = torch.einsum("bgci, cij -> bgij", query, query_embedding)
        key_embedded = torch.einsum("bgci, cij -> bgij", key, key_embedding)
        query_key = torch.einsum("bgci, bgcj -> bgij", query_embedded, key_embedded)

        similarity = torch.cat([query_key, query_embedded, key_embedded], dim=1)
        similarity = self.similarity_normalization(similarity).view(batch_size * dim_1, 3, self.groups, dim_attention, dim_attention).sum(dim=1)
        similarity = F.softmax(similarity, dim=3)

        attention_map = torch.einsum("bgij, bgcj->bgci", similarity, value)
        attention_map_embedded = torch.einsum("bgij, cij->bgci", similarity, value_embedding)
        output = torch.cat([attention_map, attention_map_embedded], dim=-1).view(batch_size * dim_1, 2 * self.out_channels, dim_attention)
        output = self.output_normalization(output).view(batch_size, dim_1, self.out_channels, 2, dim_attention).sum(dim=-2)

        return output.permute(0, 2, 3, 1) if self.dim == 0 else output.permute(0, 2, 1, 3)

class AxialAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, span, latent_vector_features=256, groups=16, activation=nn.PReLU, downscale=True, dropout=0.0):
        super(AxialAttentionBlock, self).__init__()

        span = span if isinstance(span, tuple) else (span, span)

        self.input_mapping_conv = nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), (0, 0), bias=False)
        self.input_mapping_norm = ConditionalBatchNormLayer(out_channels, latent_vector_features=latent_vector_features, track_running_stats=True)
        self.input_mapping_act = activation()

        self.axial_attention_mapping = nn.Sequential(
            AxialAttentionLayer(out_channels, out_channels, 0, span[0], groups),
            AxialAttentionLayer(out_channels, out_channels, 1, span[1], groups)
        )

        self.dropout = nn.Dropout2d(p=dropout, inplace=True)
        self.output_mapping_conv = nn.Conv2d(out_channels, out_channels, (1, 1), (1, 1), (0, 0), bias=False)
        self.output_mapping_norm = ConditionalBatchNormLayer(out_channels, latent_vector_features=latent_vector_features, track_running_stats=True)

        self.residual_mapping = nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), (0, 0), bias=False) \
            if in_channels != out_channels else nn.Identity()
        self.final_activation = activation()
        self.pooling_layer = nn.AvgPool2d((2, 2), (2, 2)) if downscale else nn.Identity()

    def forward(self, input, latent_vector):
        output = self.input_mapping_act(self.input_mapping_norm(self.input_mapping_conv(input), latent_vector))
        output = self.axial_attention_mapping(output)
        output = self.dropout(output)
        output = self.output_mapping_norm(self.output_mapping_conv(self.pooling_layer(output)), latent_vector)
        output = output + self.pooling_layer(self.residual_mapping(input))
        return self.final_activation(output)
