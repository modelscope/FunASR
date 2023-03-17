/*
 * Copyright (c) 2003, 2007-14 Matteo Frigo
 * Copyright (c) 2003, 2007-14 Massachusetts Institute of Technology
 *
 * The following statement of license applies *only* to this header file,
 * and *not* to the other files distributed with FFTW or derived therefrom:
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/***************************** NOTE TO USERS *********************************
 *
 *                 THIS IS A HEADER FILE, NOT A MANUAL
 *
 *    If you want to know how to use FFTW, please read the manual,
 *    online at http://www.fftw.org/doc/ and also included with FFTW.
 *    For a quick start, see the manual's tutorial section.
 *
 *   (Reading header files to learn how to use a library is a habit
 *    stemming from code lacking a proper manual.  Arguably, it's a
 *    *bad* habit in most cases, because header files can contain
 *    interfaces that are not part of the public, stable API.)
 *
 ****************************************************************************/

#ifndef FFTW3_H
#define FFTW3_H

#include <stdio.h>

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

/* If <complex.h> is included, use the C99 complex type.  Otherwise
   define a type bit-compatible with C99 complex */
#if !defined(FFTW_NO_Complex) && defined(_Complex_I) && defined(complex) && defined(I)
#  define FFTW_DEFINE_COMPLEX(R, C) typedef R _Complex C
#else
#  define FFTW_DEFINE_COMPLEX(R, C) typedef R C[2]
#endif

#define FFTW_CONCAT(prefix, name) prefix ## name
#define FFTW_MANGLE_DOUBLE(name) FFTW_CONCAT(fftw_, name)
#define FFTW_MANGLE_FLOAT(name) FFTW_CONCAT(fftwf_, name)
#define FFTW_MANGLE_LONG_DOUBLE(name) FFTW_CONCAT(fftwl_, name)
#define FFTW_MANGLE_QUAD(name) FFTW_CONCAT(fftwq_, name)

/* IMPORTANT: for Windows compilers, you should add a line
*/
#define FFTW_DLL
/*
   here and in kernel/ifftw.h if you are compiling/using FFTW as a
   DLL, in order to do the proper importing/exporting, or
   alternatively compile with -DFFTW_DLL or the equivalent
   command-line flag.  This is not necessary under MinGW/Cygwin, where
   libtool does the imports/exports automatically. */
#if defined(FFTW_DLL) && (defined(_WIN32) || defined(__WIN32__))
   /* annoying Windows syntax for shared-library declarations */
#  if defined(COMPILING_FFTW) /* defined in api.h when compiling FFTW */
#    define FFTW_EXTERN extern __declspec(dllexport) 
#  else /* user is calling FFTW; import symbol */
#    define FFTW_EXTERN extern __declspec(dllimport) 
#  endif
#else
#  define FFTW_EXTERN extern
#endif

enum fftw_r2r_kind_do_not_use_me {
     FFTW_R2HC=0, FFTW_HC2R=1, FFTW_DHT=2,
     FFTW_REDFT00=3, FFTW_REDFT01=4, FFTW_REDFT10=5, FFTW_REDFT11=6,
     FFTW_RODFT00=7, FFTW_RODFT01=8, FFTW_RODFT10=9, FFTW_RODFT11=10
};

struct fftw_iodim_do_not_use_me {
     int n;                     /* dimension size */
     int is;			/* input stride */
     int os;			/* output stride */
};

#include <stddef.h> /* for ptrdiff_t */
struct fftw_iodim64_do_not_use_me {
     ptrdiff_t n;                     /* dimension size */
     ptrdiff_t is;			/* input stride */
     ptrdiff_t os;			/* output stride */
};

typedef void (*fftw_write_char_func_do_not_use_me)(char c, void *);
typedef int (*fftw_read_char_func_do_not_use_me)(void *);

/*
  huge second-order macro that defines prototypes for all API
  functions.  We expand this macro for each supported precision
 
  X: name-mangling macro
  R: real data type
  C: complex data type
*/

#define FFTW_DEFINE_API(X, R, C)					   \
									   \
FFTW_DEFINE_COMPLEX(R, C);						   \
									   \
typedef struct X(plan_s) *X(plan);					   \
									   \
typedef struct fftw_iodim_do_not_use_me X(iodim);			   \
typedef struct fftw_iodim64_do_not_use_me X(iodim64);			   \
									   \
typedef enum fftw_r2r_kind_do_not_use_me X(r2r_kind);			   \
									   \
typedef fftw_write_char_func_do_not_use_me X(write_char_func);		   \
typedef fftw_read_char_func_do_not_use_me X(read_char_func);		   \
                                                                           \
FFTW_EXTERN void X(execute)(const X(plan) p);                              \
									   \
FFTW_EXTERN X(plan) X(plan_dft)(int rank, const int *n,			   \
		    C *in, C *out, int sign, unsigned flags);		   \
									   \
FFTW_EXTERN X(plan) X(plan_dft_1d)(int n, C *in, C *out, int sign,	   \
		       unsigned flags);					   \
FFTW_EXTERN X(plan) X(plan_dft_2d)(int n0, int n1,			   \
		       C *in, C *out, int sign, unsigned flags);	   \
FFTW_EXTERN X(plan) X(plan_dft_3d)(int n0, int n1, int n2,		   \
		       C *in, C *out, int sign, unsigned flags);	   \
									   \
FFTW_EXTERN X(plan) X(plan_many_dft)(int rank, const int *n,		   \
                         int howmany,					   \
                         C *in, const int *inembed,			   \
                         int istride, int idist,			   \
                         C *out, const int *onembed,			   \
                         int ostride, int odist,			   \
                         int sign, unsigned flags);			   \
									   \
FFTW_EXTERN X(plan) X(plan_guru_dft)(int rank, const X(iodim) *dims,	   \
			 int howmany_rank,				   \
			 const X(iodim) *howmany_dims,			   \
			 C *in, C *out,					   \
			 int sign, unsigned flags);			   \
FFTW_EXTERN X(plan) X(plan_guru_split_dft)(int rank, const X(iodim) *dims, \
			 int howmany_rank,				   \
			 const X(iodim) *howmany_dims,			   \
			 R *ri, R *ii, R *ro, R *io,			   \
			 unsigned flags);				   \
									   \
FFTW_EXTERN X(plan) X(plan_guru64_dft)(int rank,			   \
                         const X(iodim64) *dims,			   \
			 int howmany_rank,				   \
			 const X(iodim64) *howmany_dims,		   \
			 C *in, C *out,					   \
			 int sign, unsigned flags);			   \
FFTW_EXTERN X(plan) X(plan_guru64_split_dft)(int rank,			   \
                         const X(iodim64) *dims,			   \
			 int howmany_rank,				   \
			 const X(iodim64) *howmany_dims,		   \
			 R *ri, R *ii, R *ro, R *io,			   \
			 unsigned flags);				   \
									   \
FFTW_EXTERN void X(execute_dft)(const X(plan) p, C *in, C *out);	   \
FFTW_EXTERN void X(execute_split_dft)(const X(plan) p, R *ri, R *ii,	   \
                                      R *ro, R *io);			   \
									   \
FFTW_EXTERN X(plan) X(plan_many_dft_r2c)(int rank, const int *n,	   \
                             int howmany,				   \
                             R *in, const int *inembed,			   \
                             int istride, int idist,			   \
                             C *out, const int *onembed,		   \
                             int ostride, int odist,			   \
                             unsigned flags);				   \
									   \
FFTW_EXTERN X(plan) X(plan_dft_r2c)(int rank, const int *n,		   \
                        R *in, C *out, unsigned flags);			   \
									   \
FFTW_EXTERN X(plan) X(plan_dft_r2c_1d)(int n,R *in,C *out,unsigned flags); \
FFTW_EXTERN X(plan) X(plan_dft_r2c_2d)(int n0, int n1,			   \
			   R *in, C *out, unsigned flags);		   \
FFTW_EXTERN X(plan) X(plan_dft_r2c_3d)(int n0, int n1,			   \
			   int n2,					   \
			   R *in, C *out, unsigned flags);		   \
									   \
									   \
FFTW_EXTERN X(plan) X(plan_many_dft_c2r)(int rank, const int *n,	   \
			     int howmany,				   \
			     C *in, const int *inembed,			   \
			     int istride, int idist,			   \
			     R *out, const int *onembed,		   \
			     int ostride, int odist,			   \
			     unsigned flags);				   \
									   \
FFTW_EXTERN X(plan) X(plan_dft_c2r)(int rank, const int *n,		   \
                        C *in, R *out, unsigned flags);			   \
									   \
FFTW_EXTERN X(plan) X(plan_dft_c2r_1d)(int n,C *in,R *out,unsigned flags); \
FFTW_EXTERN X(plan) X(plan_dft_c2r_2d)(int n0, int n1,			   \
			   C *in, R *out, unsigned flags);		   \
FFTW_EXTERN X(plan) X(plan_dft_c2r_3d)(int n0, int n1,			   \
			   int n2,					   \
			   C *in, R *out, unsigned flags);		   \
									   \
FFTW_EXTERN X(plan) X(plan_guru_dft_r2c)(int rank, const X(iodim) *dims,   \
			     int howmany_rank,				   \
			     const X(iodim) *howmany_dims,		   \
			     R *in, C *out,				   \
			     unsigned flags);				   \
FFTW_EXTERN X(plan) X(plan_guru_dft_c2r)(int rank, const X(iodim) *dims,   \
			     int howmany_rank,				   \
			     const X(iodim) *howmany_dims,		   \
			     C *in, R *out,				   \
			     unsigned flags);				   \
									   \
FFTW_EXTERN X(plan) X(plan_guru_split_dft_r2c)(				   \
                             int rank, const X(iodim) *dims,		   \
			     int howmany_rank,				   \
			     const X(iodim) *howmany_dims,		   \
			     R *in, R *ro, R *io,			   \
			     unsigned flags);				   \
FFTW_EXTERN X(plan) X(plan_guru_split_dft_c2r)(				   \
                             int rank, const X(iodim) *dims,		   \
			     int howmany_rank,				   \
			     const X(iodim) *howmany_dims,		   \
			     R *ri, R *ii, R *out,			   \
			     unsigned flags);				   \
									   \
FFTW_EXTERN X(plan) X(plan_guru64_dft_r2c)(int rank,			   \
                             const X(iodim64) *dims,			   \
			     int howmany_rank,				   \
			     const X(iodim64) *howmany_dims,		   \
			     R *in, C *out,				   \
			     unsigned flags);				   \
FFTW_EXTERN X(plan) X(plan_guru64_dft_c2r)(int rank,			   \
                             const X(iodim64) *dims,			   \
			     int howmany_rank,				   \
			     const X(iodim64) *howmany_dims,		   \
			     C *in, R *out,				   \
			     unsigned flags);				   \
									   \
FFTW_EXTERN X(plan) X(plan_guru64_split_dft_r2c)(			   \
                             int rank, const X(iodim64) *dims,		   \
			     int howmany_rank,				   \
			     const X(iodim64) *howmany_dims,		   \
			     R *in, R *ro, R *io,			   \
			     unsigned flags);				   \
FFTW_EXTERN X(plan) X(plan_guru64_split_dft_c2r)(			   \
                             int rank, const X(iodim64) *dims,		   \
			     int howmany_rank,				   \
			     const X(iodim64) *howmany_dims,		   \
			     R *ri, R *ii, R *out,			   \
			     unsigned flags);				   \
									   \
FFTW_EXTERN void X(execute_dft_r2c)(const X(plan) p, R *in, C *out);	   \
FFTW_EXTERN void X(execute_dft_c2r)(const X(plan) p, C *in, R *out);	   \
									   \
FFTW_EXTERN void X(execute_split_dft_r2c)(const X(plan) p,		   \
                                          R *in, R *ro, R *io);		   \
FFTW_EXTERN void X(execute_split_dft_c2r)(const X(plan) p,		   \
                                          R *ri, R *ii, R *out);	   \
									   \
FFTW_EXTERN X(plan) X(plan_many_r2r)(int rank, const int *n,		   \
                         int howmany,					   \
                         R *in, const int *inembed,			   \
                         int istride, int idist,			   \
                         R *out, const int *onembed,			   \
                         int ostride, int odist,			   \
                         const X(r2r_kind) *kind, unsigned flags);	   \
									   \
FFTW_EXTERN X(plan) X(plan_r2r)(int rank, const int *n, R *in, R *out,	   \
                    const X(r2r_kind) *kind, unsigned flags);		   \
									   \
FFTW_EXTERN X(plan) X(plan_r2r_1d)(int n, R *in, R *out,		   \
                       X(r2r_kind) kind, unsigned flags);		   \
FFTW_EXTERN X(plan) X(plan_r2r_2d)(int n0, int n1, R *in, R *out,	   \
                       X(r2r_kind) kind0, X(r2r_kind) kind1,		   \
                       unsigned flags);					   \
FFTW_EXTERN X(plan) X(plan_r2r_3d)(int n0, int n1, int n2,		   \
                       R *in, R *out, X(r2r_kind) kind0,		   \
                       X(r2r_kind) kind1, X(r2r_kind) kind2,		   \
                       unsigned flags);					   \
									   \
FFTW_EXTERN X(plan) X(plan_guru_r2r)(int rank, const X(iodim) *dims,	   \
                         int howmany_rank,				   \
                         const X(iodim) *howmany_dims,			   \
                         R *in, R *out,					   \
                         const X(r2r_kind) *kind, unsigned flags);	   \
									   \
FFTW_EXTERN X(plan) X(plan_guru64_r2r)(int rank, const X(iodim64) *dims,   \
                         int howmany_rank,				   \
                         const X(iodim64) *howmany_dims,		   \
                         R *in, R *out,					   \
                         const X(r2r_kind) *kind, unsigned flags);	   \
									   \
FFTW_EXTERN void X(execute_r2r)(const X(plan) p, R *in, R *out);	   \
									   \
FFTW_EXTERN void X(destroy_plan)(X(plan) p);				   \
FFTW_EXTERN void X(forget_wisdom)(void);				   \
FFTW_EXTERN void X(cleanup)(void);					   \
									   \
FFTW_EXTERN void X(set_timelimit)(double t);				   \
									   \
FFTW_EXTERN void X(plan_with_nthreads)(int nthreads);			   \
FFTW_EXTERN int X(init_threads)(void);					   \
FFTW_EXTERN void X(cleanup_threads)(void);				   \
FFTW_EXTERN void X(make_planner_thread_safe)(void);                        \
									   \
FFTW_EXTERN int X(export_wisdom_to_filename)(const char *filename);	   \
FFTW_EXTERN void X(export_wisdom_to_file)(FILE *output_file);		   \
FFTW_EXTERN char *X(export_wisdom_to_string)(void);			   \
FFTW_EXTERN void X(export_wisdom)(X(write_char_func) write_char,   	   \
                                  void *data);				   \
FFTW_EXTERN int X(import_system_wisdom)(void);				   \
FFTW_EXTERN int X(import_wisdom_from_filename)(const char *filename);	   \
FFTW_EXTERN int X(import_wisdom_from_file)(FILE *input_file);		   \
FFTW_EXTERN int X(import_wisdom_from_string)(const char *input_string);	   \
FFTW_EXTERN int X(import_wisdom)(X(read_char_func) read_char, void *data); \
									   \
FFTW_EXTERN void X(fprint_plan)(const X(plan) p, FILE *output_file);	   \
FFTW_EXTERN void X(print_plan)(const X(plan) p);			   \
FFTW_EXTERN char *X(sprint_plan)(const X(plan) p);			   \
									   \
FFTW_EXTERN void *X(malloc)(size_t n);					   \
FFTW_EXTERN R *X(alloc_real)(size_t n);					   \
FFTW_EXTERN C *X(alloc_complex)(size_t n);				   \
FFTW_EXTERN void X(free)(void *p);					   \
									   \
FFTW_EXTERN void X(flops)(const X(plan) p,				   \
                          double *add, double *mul, double *fmas);	   \
FFTW_EXTERN double X(estimate_cost)(const X(plan) p);			   \
FFTW_EXTERN double X(cost)(const X(plan) p);				   \
									   \
FFTW_EXTERN int X(alignment_of)(R *p);                                     \
FFTW_EXTERN const char X(version)[];                                       \
FFTW_EXTERN const char X(cc)[];						   \
FFTW_EXTERN const char X(codelet_optim)[];


/* end of FFTW_DEFINE_API macro */

FFTW_DEFINE_API(FFTW_MANGLE_DOUBLE, double, fftw_complex)
FFTW_DEFINE_API(FFTW_MANGLE_FLOAT, float, fftwf_complex)
FFTW_DEFINE_API(FFTW_MANGLE_LONG_DOUBLE, long double, fftwl_complex)

/* __float128 (quad precision) is a gcc extension on i386, x86_64, and ia64
   for gcc >= 4.6 (compiled in FFTW with --enable-quad-precision) */
#if (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)) \
 && !(defined(__ICC) || defined(__INTEL_COMPILER) || defined(__CUDACC__) || defined(__PGI)) \
 && (defined(__i386__) || defined(__x86_64__) || defined(__ia64__))
#  if !defined(FFTW_NO_Complex) && defined(_Complex_I) && defined(complex) && defined(I)
/* note: __float128 is a typedef, which is not supported with the _Complex
         keyword in gcc, so instead we use this ugly __attribute__ version.
         However, we can't simply pass the __attribute__ version to
         FFTW_DEFINE_API because the __attribute__ confuses gcc in pointer
         types.  Hence redefining FFTW_DEFINE_COMPLEX.  Ugh. */
#    undef FFTW_DEFINE_COMPLEX
#    define FFTW_DEFINE_COMPLEX(R, C) typedef _Complex float __attribute__((mode(TC))) C
#  endif
FFTW_DEFINE_API(FFTW_MANGLE_QUAD, __float128, fftwq_complex)
#endif

#define FFTW_FORWARD (-1)
#define FFTW_BACKWARD (+1)

#define FFTW_NO_TIMELIMIT (-1.0)

/* documented flags */
#define FFTW_MEASURE (0U)
#define FFTW_DESTROY_INPUT (1U << 0)
#define FFTW_UNALIGNED (1U << 1)
#define FFTW_CONSERVE_MEMORY (1U << 2)
#define FFTW_EXHAUSTIVE (1U << 3) /* NO_EXHAUSTIVE is default */
#define FFTW_PRESERVE_INPUT (1U << 4) /* cancels FFTW_DESTROY_INPUT */
#define FFTW_PATIENT (1U << 5) /* IMPATIENT is default */
#define FFTW_ESTIMATE (1U << 6)
#define FFTW_WISDOM_ONLY (1U << 21)

/* undocumented beyond-guru flags */
#define FFTW_ESTIMATE_PATIENT (1U << 7)
#define FFTW_BELIEVE_PCOST (1U << 8)
#define FFTW_NO_DFT_R2HC (1U << 9)
#define FFTW_NO_NONTHREADED (1U << 10)
#define FFTW_NO_BUFFERING (1U << 11)
#define FFTW_NO_INDIRECT_OP (1U << 12)
#define FFTW_ALLOW_LARGE_GENERIC (1U << 13) /* NO_LARGE_GENERIC is default */
#define FFTW_NO_RANK_SPLITS (1U << 14)
#define FFTW_NO_VRANK_SPLITS (1U << 15)
#define FFTW_NO_VRECURSE (1U << 16)
#define FFTW_NO_SIMD (1U << 17)
#define FFTW_NO_SLOW (1U << 18)
#define FFTW_NO_FIXED_RADIX_LARGE_N (1U << 19)
#define FFTW_ALLOW_PRUNING (1U << 20)

#ifdef __cplusplus
}  /* extern "C" */
#endif /* __cplusplus */

#endif /* FFTW3_H */
