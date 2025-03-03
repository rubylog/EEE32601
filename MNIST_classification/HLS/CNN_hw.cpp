#include "dma_template.h"

int8_t popcount9(uint16_t value, const int8_t* popcount_lut_9) {
    return popcount_lut_9[value];
}

int8_t popcount8(uint16_t value, const int8_t* popcount_lut_8) {
    return popcount_lut_8[value];
}

void load_input(AXI_VAL_uint8* input_stream, 
                     uint8_t input_image[BATCH_SIZE][28][28]) {

    for (int t = 0; t < BATCH_SIZE; t++) {
        // t is each image
        for (int row = 0; row < 28; row++) {
            for (int col = 0; col < 28; col++) {
                int idx = t * 28 * 28 + row * 28 + col;
                input_image[t][row][col] = pop_stream<uint8_t>(input_stream[idx]);
            }
        }
    }
}

void pre_processing(uint8_t input_image[BATCH_SIZE][28][28], 
                    uint16_t input_layer1[BATCH_SIZE][26][26]) {
    for (int t = 0; t < BATCH_SIZE; t++) {
        for (int row = 0; row < 26; row++) {
            for (int col = 0; col < 26; col++) {
                uint16_t packedValue = 0;

                packedValue |= (input_image[t][row][col] & 1) << 8;
                packedValue |= (input_image[t][row][col + 1] & 1) << 7;
                packedValue |= (input_image[t][row][col + 2] & 1) << 6;
                packedValue |= (input_image[t][row + 1][col] & 1) << 5;
                packedValue |= (input_image[t][row + 1][col + 1] & 1) << 4;
                packedValue |= (input_image[t][row + 1][col + 2] & 1) << 3;
                packedValue |= (input_image[t][row + 2][col] & 1) << 2;
                packedValue |= (input_image[t][row + 2][col + 1] & 1) << 1;
                packedValue |= (input_image[t][row + 2][col + 2] & 1);

                input_layer1[t][row][col] = packedValue;
            }
        }
    }
}

void layer1_conv2d(uint16_t input_layer1[BATCH_SIZE][26][26], 
                             uint8_t output_layer1[BATCH_SIZE][16][26][26]) {

    for (int t = 0; t < BATCH_SIZE; t++) {
        for (int nk = 0; nk < 16; nk++) {
            for (int row = 0; row < 26; row++) {
                for (int col = 0; col < 26; col++) {
#pragma HLS PIPELINE II=1
                    int16_t popcount_result_all_channel = 0;
                    uint16_t xnor_result_each_channel = 
                        ~(conv1_kernels[nk] ^ input_layer1[t][row][col]);

                    popcount_result_all_channel = popcount9(xnor_result_each_channel, popcount_lut_9);

                    output_layer1[t][nk][row][col] = (popcount_result_all_channel >= -7) ? 1 : 0;
                    // without bit packing
                    // activation function line (instead of relu)
                }
            }
        }
    }
}

void layer1_packing(uint8_t output_layer1[BATCH_SIZE][16][26][26], 
                    uint16_t input_layer2[BATCH_SIZE][16][24][24]) {

    for (int t = 0; t < BATCH_SIZE; t++) {
        for (int nk = 0; nk < 16; nk++) {
            for (int row = 0; row < 24; row++) {
                for (int col = 0; col < 24; col++) {
#pragma HLS PIPELINE II=1
                    uint16_t packedValue = 0;

                    packedValue |= (output_layer1[t][nk][row][col] & 1) << 8;
                    packedValue |= (output_layer1[t][nk][row][col + 1] & 1) << 7;
                    packedValue |= (output_layer1[t][nk][row][col + 2] & 1) << 6;
                    packedValue |= (output_layer1[t][nk][row + 1][col] & 1) << 5;
                    packedValue |= (output_layer1[t][nk][row + 1][col + 1] & 1) << 4;
                    packedValue |= (output_layer1[t][nk][row + 1][col + 2] & 1) << 3;
                    packedValue |= (output_layer1[t][nk][row + 2][col] & 1) << 2;
                    packedValue |= (output_layer1[t][nk][row + 2][col + 1] & 1) << 1;
                    packedValue |= (output_layer1[t][nk][row + 2][col + 2] & 1);

                    input_layer2[t][nk][row][col] = packedValue;
                }
            }
        }
    }
}

// 16 = OUTPUT_CHANNEL_SIZE

void layer2_conv2d(uint16_t input_layer2[BATCH_SIZE][16][24][24], 
                uint8_t flatten[BATCH_SIZE][16 * 12 * 12]) {

    uint8_t output_layer2[BATCH_SIZE][16][24][24];
    // #pragma HLS INLINE region
    for (int t = 0; t < BATCH_SIZE; t++) {
        int flatten_idx = 0;
        for (int nk = 0; nk < 16; nk++) {

            for (int row = 0; row < 24; row++) {
                for (int col = 0; col < 24; col++) {
                    int16_t popcount_result_all_channel = 0;
                    for (int c = 0; c < 16 ; c++) {
                        #pragma HLS UNROLL factor=16
                        uint16_t xnor_result_each_channel = 
                            ~(conv2_kernels[c][nk] ^ input_layer2[t][c][row][col]);
                        popcount_result_all_channel += popcount9(xnor_result_each_channel, popcount_lut_9);
                    }
                    output_layer2[t][nk][row][col] = (popcount_result_all_channel >= -7*16) ? 1 : 0; // activation
                    // without bit packing
                }
            }

           for (int row = 0; row < 12; row++) {
                for (int col = 0; col < 12; col++) {
                    #pragma HLS PIPELINE II=1
                    int base_row = row * 2;
                    int base_col = col * 2;

                    uint8_t sum_val = 
                        output_layer2[t][nk][base_row][base_col]
                        + output_layer2[t][nk][base_row][base_col + 1]
                        + output_layer2[t][nk][base_row + 1][base_col]
                        + output_layer2[t][nk][base_row + 1][base_col + 1];

                    flatten[t][flatten_idx++] = (sum_val >= 2);
                }
            }
        }
    }
}

void pack_flatten_output(uint8_t flatten[BATCH_SIZE][16 * 12 * 12], 
                         uint8_t packed_output[BATCH_SIZE][2 * 12 * 12]) {
    for (int t = 0; t < BATCH_SIZE; t++) {
        int packed_idx = 0;      
        uint8_t current_byte = 0; 

        for (int i = 0; i < 16 * 12 * 12; i++) {
            current_byte |= (flatten[t][i] & 1) << (7 - (i % 8));

            if ((i + 1) % 8 == 0) {
                packed_output[t][packed_idx++] = current_byte;
                current_byte = 0;
            }
        }
    }
}

void layer3_fc(uint8_t packed_output[BATCH_SIZE][2 * 12 * 12],
                int16_t fc_output[BATCH_SIZE][10]){
    for (int t = 0; t < BATCH_SIZE; t++) { 
        for (int out_idx = 0; out_idx < 10; out_idx++) { 
            int16_t sum = 0; 
            for (int i = 0; i < 288; i++) { 
                uint8_t xnor_result = ~(packed_output[t][i] ^ fc_weights[out_idx][i]);
                sum += popcount8(xnor_result, popcount_lut_8);
            }
            fc_output[t][out_idx] = sum;
        }
    }
}

void store_output(AXI_VAL_int16* output_stream, 
                       int16_t fc_output[BATCH_SIZE][10]) {

    for (int t = 0; t < BATCH_SIZE; t++) {
        for (int k = 0; k < 10; k++) {
            int idx = t * 10 + k; 
            bool is_last = (t == BATCH_SIZE - 1) && (k == 9);
            output_stream[idx] = push_stream<int16_t>(fc_output[t][k], is_last);
        }
    }
}

// Define Top Function
void top_function(AXI_VAL_uint8* input_stream,
                  AXI_VAL_int16* output_stream) {
    #pragma HLS INTERFACE axis port=input_stream
    #pragma HLS INTERFACE axis port=output_stream
	#pragma HLS INTERFACE ap_ctrl_none port=return

    uint8_t input_image[BATCH_SIZE][28][28];
    uint16_t input_layer1[BATCH_SIZE][26][26];
    uint8_t output_layer1[BATCH_SIZE][16][26][26];
    uint16_t input_layer2[BATCH_SIZE][16][24][24];
    uint8_t flatten[BATCH_SIZE][16 * 12 * 12];
    uint8_t packed_output[BATCH_SIZE][2 * 12 * 12];
    int16_t fc_output[BATCH_SIZE][10];

#pragma HLS ARRAY_PARTITION variable=conv2_kernels complete dim=1  // for all channel
#pragma HLS ARRAY_PARTITION variable=input_layer2 complete dim=1 

#pragma HLS ARRAY_PARTITION variable=output_layer1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=input_layer2 complete dim=2

#pragma HLS ARRAY_PARTITION variable=input_layer1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=output_layer1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=conv1_kernels complete dim=1


    // Process each tile

    // 1. Load Input Tile
    load_input(input_stream, input_image);

    // 2. Pre-processing input_image
    pre_processing(input_image, input_layer1);

    // 3. Layer 1: Convolution
    layer1_conv2d(input_layer1, output_layer1);

    // 4. Layer 1: Packing
    layer1_packing(output_layer1, input_layer2);

    // 5. Layer 2: Convolution + Avg Pooling
    layer2_conv2d(input_layer2, flatten);

    // 6. Pack Flattened Output
    pack_flatten_output(flatten, packed_output);

    // 7. Fully Connected Layer
    layer3_fc(packed_output, fc_output);

    // 8. Store Output Tile
    store_output(output_stream, fc_output);

}
