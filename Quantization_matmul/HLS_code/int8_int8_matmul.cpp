#include "int8_int8_matmul.h"
const int ROW_A = 64;
const int COL_A = 128;
const int ROW_B = 128;
const int COL_B = 64;

static void load_input(AXI_VAL* in, __int8* buffer, int size) {
    for (int i = 0; i < size; i++) {
    #pragma HLS PIPELINE II=1
        buffer[i] = pop_stream<__int8>(in[i]);
    }
}

static void compute_mul(__int8* in1_buffer, __int8* in2_buffer, __int8* out_buffer) {
    __int8 A[ROW_A][COL_A];
    __int8 B[ROW_B][COL_B];
    __int8 C[ROW_A][COL_B] = {0};

    // Load A from in1_buffer
    for (int i = 0; i < ROW_A; i++) {
        for (int j = 0; j < COL_A; j++) {
        #pragma HLS PIPELINE II=1
            A[i][j] = in1_buffer[i * COL_A + j];
        }
    }

    // Load B from in2_buffer
    for (int i = 0; i < ROW_B; i++) {
        for (int j = 0; j < COL_B; j++) {
        #pragma HLS PIPELINE II=1
            B[i][j] = in2_buffer[i * COL_B + j];
        }
    }

    // Matrix multiplication
    for (int i = 0; i < ROW_A; i++) {
        for (int j = 0; j < COL_B; j++) {
            __int16 sum = 0;
            for (int k = 0; k < COL_A; k++) {
            #pragma HLS PIPELINE II=1
                sum += A[i][k] * B[k][j];
            }
            if(sum > 127)
                sum = 127;
            else if (sum < -128) 
                sum = 128;
            // above logics are clamping 16bit result to 8 bit.
            C[i][j] = sum;
        }
    }

    // Store result in out_buffer
    for (int i = 0; i < ROW_A; i++) {
        for (int j = 0; j < COL_B; j++) {
        #pragma HLS PIPELINE II=1
            out_buffer[i * COL_B + j] = C[i][j];
        }
    }
}

static void store_result(AXI_VAL* out, __int8* buffer, int size) {
    for (int i = 0; i < size; i++) {
    #pragma HLS PIPELINE II=1
        out[i] = push_stream<__int8>(buffer[i], i == size - 1);
    }
}

void matmul_hw(AXI_VAL* in1, AXI_VAL* in2, AXI_VAL* out) {
#pragma HLS INTERFACE axis port=in1 depth=64*128
#pragma HLS INTERFACE axis port=in2 depth=128*64
#pragma HLS INTERFACE axis port=out depth=64*64
#pragma HLS INTERFACE ap_ctrl_none port=return

    __int8 in1_buffer[ROW_A * COL_A];
    __int8 in2_buffer[ROW_B * COL_B];
    __int8 out_buffer[ROW_A * COL_B];

#pragma HLS DATAFLOW

    load_input(in1, in1_buffer, ROW_A * COL_A);
    load_input(in2, in2_buffer, ROW_B * COL_B);
    compute_mul(in1_buffer, in2_buffer, out_buffer);
    store_result(out, out_buffer, ROW_A * COL_B);
}
