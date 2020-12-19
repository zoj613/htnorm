#ifndef HTNORM_ALWAYS_INLINE_H
#define HTNORM_ALWAYS_INLINE_H

#if defined(__GNUC__)
    #define ALWAYS_INLINE(ret_type) inline ret_type __attribute__((always_inline))
#elif defined(__clang__)
    #if __has_attribute(always_inline)
        #define ALWAYS_INLINE(ret_type) inline ret_type __attribute__((always_inline))
    #else
        #define ALWAYS_INLINE(ret_type) inline ret_type
    #endif
#elif defined(_MSC_VER)
    #define ALWAYS_INLINE(ret_type) __forceinline ret_type
#else
    #define ALWAYS_INLINE(ret_type) inline ret_type
#endif

#endif
