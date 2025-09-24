#ifndef GOLDEN
#define GOLDEN

/**
 * Input:
 *   A : [num_rows][num_cols]ElTp
 * Result:
 *   B : [num_rows][num_cols]ElTp
 */
template<class ElTp>
void goldenSeq(ElTp* A, ElTp* B, const uint32_t num_rows, const uint32_t num_cols) {
    /**************************************************/
    /*** CUDA exercise 2 subtask 1:                 ***/
    /***   Please parallelize correctly the outer   ***/
    /***   loop of count `i` by inserting:          ***/
    /***   - an OpenMP pragma                       ***/
    /***   - and if neccessary by a very tiny bit   ***/
    /***     of code changes                        ***/
    /**************************************************/
    ElTp accum, a_el;
    
    for(uint64_t i = 0; i < num_rows; i++) {
        uint64_t ii = i*num_cols;
        accum = 0.0;
        for(uint64_t j = 0; j < num_cols; j++) {
            a_el  = A[ii + j];
            accum = sqrt(accum) + a_el*a_el;
            B[ii + j] = accum;
        }
    }
}

#endif
