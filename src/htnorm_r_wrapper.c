/* Copyright (c) 2020, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#include <R.h>
#include <Rinternals.h>

#include "htnorm.h"

enum RNG_TYPE {Xoroshiro128p, PCG64}; 

static void
finalize_rng_xp(SEXP rng_xp)
{
   // free the rng external pointer when no longer needed
   rng_t* rng = R_ExternalPtrAddr(rng_xp);
   rng_free(rng); 
}

/* create an external pointer to wrap the rng_t struct */
static SEXP
new_rng_extptr(SEXP seed, SEXP gen)
{
    rng_t* rng;
    int cgen = *INTEGER(gen);

    if (cgen == PCG64 && (seed != R_NilValue)) {
        rng = rng_pcg64_new_seeded(*INTEGER(seed));
    }
    else if (cgen == PCG64) {
        rng = rng_pcg64_new();
    }
    else if (seed != R_NilValue) {
        rng = rng_xrs128p_new_seeded(*INTEGER(seed));
    }
    else {
        rng = rng_xrs128p_new();
    }

    SEXP rng_xp = R_MakeExternalPtr(rng, R_NilValue, R_NilValue);

    R_RegisterCFinalizerEx(rng_xp, finalize_rng_xp, TRUE);
   
    return rng_xp;
}


static SEXP
hp_trunc_mvn(SEXP rng_xp, SEXP mean, SEXP cov, SEXP g, SEXP r, SEXP diag, SEXP out)
{
    ht_config_t config;
    SEXP info, list;
    int protect_stack_count = 2;
    rng_t* rng = R_ExternalPtrAddr(rng_xp);

    init_ht_config(&config, (size_t)nrows(g), (size_t)ncols(g), REAL(mean),
                   REAL(cov), REAL(g), REAL(r), *LOGICAL(diag));
    
    const char* names[] = {"info", "out", ""};
    list = PROTECT(Rf_mkNamed(VECSXP, names));
    info = PROTECT(allocVector(INTSXP, 1));

    if (out == R_NilValue) {
        out = PROTECT(allocVector(REALSXP, config.gncol));
        protect_stack_count++;
    }

    *INTEGER(info) = htn_hyperplane_truncated_mvn(rng, &config, REAL(out));
    SET_VECTOR_ELT(list, 0, info);
    SET_VECTOR_ELT(list, 1, out);

    UNPROTECT(protect_stack_count);
    return list;
}


static SEXP
struct_prec_mvn(SEXP rng_xp, SEXP mean, SEXP a, SEXP phi, SEXP omega,
                SEXP str_mean, SEXP a_id, SEXP o_id, SEXP out)
{
    sp_config_t config;
    SEXP info, list;
    int protect_stack_count = 2;
    rng_t* rng = R_ExternalPtrAddr(rng_xp);

    init_sp_config(&config, (size_t)nrows(phi), (size_t)ncols(phi), REAL(mean),
                   REAL(a), REAL(phi), REAL(omega), *LOGICAL(str_mean),
                   *INTEGER(a_id), *INTEGER(o_id));

    const char* names[] = {"info", "out", ""};
    list = PROTECT(Rf_mkNamed(VECSXP, names));
    info = PROTECT(allocVector(INTSXP, 1));

    if (out == R_NilValue) {
        out = PROTECT(allocVector(REALSXP, config.pncol));
        protect_stack_count++;
    }

    *INTEGER(info) = htn_structured_precision_mvn(rng, &config, REAL(out));
    SET_VECTOR_ELT(list, 0, info);
    SET_VECTOR_ELT(list, 1, out);

    UNPROTECT(protect_stack_count);
    return list;
}


static const R_CallMethodDef htnorm_c_funcs[] = {
    {"get_rng", (DL_FUNC)&new_rng_extptr, 2},
    {"hpmvn", (DL_FUNC)&hp_trunc_mvn, 7},
    {"spmvn", (DL_FUNC)&struct_prec_mvn, 9},
    {NULL, NULL, 0}
};


void
R_init_htnorm(DllInfo* info)
{
    R_registerRoutines(info, NULL, htnorm_c_funcs, NULL, NULL);
    R_useDynamicSymbols(info, FALSE);
}
