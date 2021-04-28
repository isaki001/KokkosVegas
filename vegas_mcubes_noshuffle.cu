/*

code works for gaussian and sin using switch statement. device pointerr/template slow
down the code by 2x

chunksize needs to be tuned based on the ncalls. For now hardwired using a switch statement


nvcc -O2 -DCUSTOM -o cudavegas ../vegas_mcubes_noshuffle.cu -arch=sm_70
OR
nvcc -O2 -DCURAND -o cudavegas ../vegas_mcubes_noshuffle.cu -arch=sm_70

example run command

nvprof ./cudavegas 0 6 0.0  10.0  2.0E+09  58, 0, 0     || correct answer is ~-49       //58 is total iters, the number after that is iters spent in the first kernel, compiling with cmake messes it up

nvprof  ./cudavegas 1 9 -1.0  1.0  1.0E+07 15 10 10

nvprof ./cudavegas 2 2 -1.0 1.0  1.0E+09 1 0 0

Last three arguments are: total iterations, iteration

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>
#include <stdint.h>
#include <ctime>

#define WARP_SIZE 32
#define BLOCK_DIM_X 128
#define ALPH 1.5
#define NDMX  500
#define MXDIM 20

#define NDMX1 NDMX+1
#define MXDIM1 MXDIM+1
#define PI 3.14159265358979323846

#define IMAX(a,b) \
    ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
      _a > _b ? _a : _b; })

#define IMIN(a,b) \
    ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
      _a < _b ? _a : _b; })


//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                          \
        cudaError_t e=cudaGetLastError();                                 \
        if(e!=cudaSuccess) {                                              \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
}

#include "func.cuh"

__inline__ __device__  void get_indx(int ms, int *da, int ND, int NINTV) {
	int dp[MXDIM];
	int j, t0, t1;
	int m = ms;
	dp[0] = 1;
	dp[1] = NINTV;


	for (j = 0; j < ND - 2; j++) {
		dp[j + 2] = dp[j + 1] * NINTV;
	}
	//
	for (j = 0; j < ND; j++) {
		t0 = dp[ND - j - 1];
		t1 = m / t0;
		da[j] = 1 + t1;
		m = m - t1 * t0;

	}
}

__global__ void vegas_kernel(int ng, int ndim, int npg, double xjac, double dxg,
                             double *result_dev, double xnd, double *xi,
                             double *d, double *dx, double *regn, int ncubes,
                             int iter, double sc, double sci, double ing,
                             int chunkSize, uint32_t totalNumThreads,
                             int LastChunk, int fcode) {
    
	__shared__ double sh_buff_fb[BLOCK_DIM_X];
	__shared__ double sh_buff_f2b[BLOCK_DIM_X];

#ifdef CUSTOM
	uint64_t temp;
	uint32_t a = 1103515245;
	uint32_t c = 12345;
	uint32_t one, expi;
	one = 1;
	expi = 31;
	uint32_t p = one << expi;
#endif

	uint32_t seed, seed_init;
	seed_init = (iter) * ncubes;
	int m = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = threadIdx.x;
	
	double fb, f2b, wgt, xn, xo, rc, f, f2, ran00;
	int kg[MXDIM + 1];
	int ia[MXDIM + 1];
	double x[MXDIM + 1];
	int k, j;
	double fbg, f2bg;
	//if(tx == 30 && blockIdx.x == 6771) printf("here m is %d\n", m);
	if(m == 0)
		printf("totalNumThreads:%lu\n", totalNumThreads);
	if (m < totalNumThreads) {
		if (m == totalNumThreads - 1) 
			chunkSize = LastChunk + 1;
		//if(tx == 30 && blockIdx.x == 6771) printf("here m is %d\n", m);
		seed = seed_init + m * chunkSize;
#ifdef CURAND
		curandState localState;
		curand_init(seed, 0, 0, &localState);
#endif
		fbg = f2bg = 0.0;
		get_indx(m * chunkSize, &kg[1], ndim, ng);
		
		for (int t = 0; t < chunkSize; t++) {
			fb = f2b = 0.0;
			//get_indx(m * chunkSize + t, &kg[1], ndim, ng);

			for ( k = 1; k <= npg; k++) {
				wgt = xjac;

				for ( j = 1; j <= ndim; j++) {
#ifdef CUSTOM
					temp =  a * seed + c;
					seed = temp & (p - 1);
					ran00 = (double) seed / (double) p ;
#endif
#ifdef CURAND
					ran00 = curand_uniform(&localState);
#endif

					xn = (kg[j] - ran00) * dxg + 1.0;
					ia[j] = IMAX(IMIN((int)(xn), NDMX), 1);

                    if(m == 345)
						printf("iaj:%i dim%i bin:(%.15f, %.15f) chunk:%i npg:%i bin index:%i\n", 
                            ia[j], 
                            j,  
                            xi[j * (NDMX + 1) + ia[j] - 1], 
                            xi[j * (NDMX + 1) + ia[j]], 
                            t, 
                            k,
							j * NDMX1 + ia[j] - 1);

					if (ia[j] > 1) {
						xo = xi[j * NDMX1 + ia[j]] - xi[j * NDMX1 + ia[j] - 1];
						rc = xi[j * NDMX1 + ia[j] - 1] + (xn - ia[j]) * xo;
					} else {
						xo = xi[j * NDMX1 + ia[j]];
						rc = (xn - ia[j]) * xo;
					}
					
                            
					x[j] = regn[j] + rc * dx[j];

					wgt *= xo * xnd;
				}
				//double tmp = func[1](x, ndim);
				double tmp;

				switch (fcode) {
				case 0:
					tmp = (*func0)(x, ndim);
					break;
				case 1:
					tmp = (*func1)(x, ndim);
					break;
				case 2:
					tmp = (*func2)(x, ndim);
					break;
				default:
					tmp = (*func0)(x, ndim);
					break;
				}

//        tmp = (*func2)(x, ndim);
				f = wgt * tmp;
				f2 = f * f;

				fb += f;
				f2b += f2;
#pragma unroll 2
				for ( j = 1; j <= ndim; j++) {
					atomicAdd(&d[ia[j]*MXDIM1 + j], fabs(f));
					//d[ia[j]*MXDIM1 + j] += f2;
				}

			}  // end of npg loop

			f2b = sqrt(f2b * npg);
			f2b = (f2b - fb) * (f2b + fb);

			fbg += fb;
			f2bg += f2b;

			for (int k = ndim; k >= 1; k--) {
				kg[k] %= ng;
				if (++kg[k] != 1) 
					break;
			}

		} //end of chunk for loop

		//	fbg  = blockReduceSum(fbg);
		//	f2bg = blockReduceSum(f2bg);

		sh_buff_fb[tx] = fbg;
		sh_buff_f2b[tx] = f2bg;

		__syncthreads();

		if (tx == 0) {
			double fbgs = 0.0;
			double f2bgs = 0.0;
			for (int ii = 0; ii < BLOCK_DIM_X; ++ii)
			{
				fbgs += sh_buff_fb[ii];
				f2bgs += sh_buff_f2b[ii];
			}
			atomicAdd(&result_dev[0], fbgs);
			atomicAdd(&result_dev[1], f2bgs);
		}


	} // end of subcube if

}

__global__ void vegas_kernelF(int ng, int ndim, int npg, double xjac, double dxg,
                              double *result_dev, double xnd, double *xi,
                              double *d, double *dx, double *regn, int ncubes,
                              int iter, double sc, double sci, double ing,
                              int chunkSize, uint32_t totalNumThreads,
                              int LastChunk, int fcode) {

	__shared__ double sh_buff_fb[BLOCK_DIM_X];
	__shared__ double sh_buff_f2b[BLOCK_DIM_X];

#ifdef CUSTOM
	uint64_t temp;
	uint32_t a = 1103515245;
	uint32_t c = 12345;
	uint32_t one, expi;
	one = 1;
	expi = 31;
	uint32_t p = one << expi;
#endif


	uint32_t seed, seed_init;
	seed_init = (iter) * ncubes;

	int m = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = threadIdx.x;

	double fb, f2b, wgt, xn, xo, rc, f, f2, ran00;
	int kg[MXDIM + 1];
	//int ia[MXDIM + 1];
	int iaj;
	double x[MXDIM + 1];
	int k, j;
	double fbg, f2bg;
	//if(tx == 30 && blockIdx.x == 6771) printf("here m is %d\n", m);

	if (m < totalNumThreads) {
		if (m == totalNumThreads - 1) 
			chunkSize = LastChunk + 1;
		//if(tx == 30 && blockIdx.x == 6771) printf("here m is %d\n", m);
		seed = seed_init + m * chunkSize;
#ifdef CURAND
		curandState localState;
		curand_init(seed, 0, 0, &localState);
#endif
		fbg = f2bg = 0.0;
		get_indx(m * chunkSize, &kg[1], ndim, ng);
		for (int t = 0; t < chunkSize; t++) {
			fb = f2b = 0.0;
			//get_indx(m * chunkSize + t, &kg[1], ndim, ng);

			for ( k = 1; k <= npg; k++) {
				wgt = xjac;

				for ( j = 1; j <= ndim; j++) {
#ifdef CUSTOM
					temp =  a * seed + c;
					seed = temp & (p - 1);
					ran00 = (double) seed / (double) p ;
#endif
#ifdef CURAND
					ran00 = curand_uniform(&localState);
#endif

					xn = (kg[j] - ran00) * dxg + 1.0;
					iaj = IMAX(IMIN((int)(xn), NDMX), 1);
                    
                    if(m == 345)
                            printf("iaj:%i dim%i bin:(%.15f, %.15f) chunk:%i npg:%i bin index:%i\n", iaj, 
                                j,  
                                xi[j * NDMX1 + iaj - 1], 
                                xi[j * NDMX1 + iaj], 
                                t, 
                                k,
                                j * (NDMX + 1) + iaj - 1);
                    
					if (iaj > 1) {
						xo = xi[j * NDMX1 + iaj] - xi[j * NDMX1 + iaj - 1];
						rc = xi[j * NDMX1 + iaj - 1] + (xn - iaj) * xo;
					} else {
						xo = xi[j * NDMX1 + iaj];
						rc = (xn - iaj) * xo;
					}

					//x[j] = regn[j] + rc * dx[j];

					x[j] = regn[1] + rc * dx[1];

					wgt *= xo * xnd;


				}
				//double tmp = func[1](x, ndim);
				double tmp;

				switch (fcode) {
				case 0:
					tmp = (*func0)(x, ndim);
					break;
				case 1:
					tmp = (*func2)(x, ndim);
					break;
				case 2:
					tmp = (*func3)(x, ndim);
					break;
				default:
					tmp = (*func2)(x, ndim);
					break;
				}

//        tmp = (*func2)(x, ndim);
				f = wgt * tmp;
				f2 = f * f;

				fb += f;
				f2b += f2;


			}  // end of npg loop

			f2b = sqrt(f2b * npg);
			f2b = (f2b - fb) * (f2b + fb);

			fbg += fb;
			f2bg += f2b;

			for (int k = ndim; k >= 1; k--) {
				kg[k] %= ng;
				if (++kg[k] != 1) 
					break;
			}

		} //end of chunk for loop

//		fbg  = blockReduceSum(fbg);
//		f2bg = blockReduceSum(f2bg);

		sh_buff_fb[tx] = fbg;
		sh_buff_f2b[tx] = f2bg;

		__syncthreads();

		if (tx == 0) {
			double fbgs = 0.0;
			double f2bgs = 0.0;
			for (int ii = 0; ii < BLOCK_DIM_X; ++ii)
			{
				fbgs += sh_buff_fb[ii];
				f2bgs += sh_buff_f2b[ii];
			}
			atomicAdd(&result_dev[0], fbgs);
			atomicAdd(&result_dev[1], f2bgs);
		}
	} // end of subcube if
}

void rebin(double rc, int nd, double r[], double xin[], double xi[])
{
	int i, k = 0;
	double dr = 0.0, xn = 0.0, xo = 0.0;
	for (i = 1; i < nd; i++) {
		while (rc > dr)
			dr += r[++k];
		if (k > 1) xo = xi[k - 1];
		xn = xi[k];
		dr -= rc;
		xin[i] = xn - (xn - xo) * dr / r[k];
	}

	for (i = 1; i < nd; i++){ 
		printf("setting x[%i] for bin %i:%.15e\n", i, i,  xin[i]);
		xi[i] = xin[i];
	}
	xi[nd] = 1.0;
}

void vegas(double regn[], int ndim, int fcode,
           double ncall, double *tgral, double *sd,
           double *chi2a, int titer, int itmax, int skip)
{
	int i, it, j, k, nd, ndo, ng, npg, ncubes;
	double calls, dv2g, dxg, rc, ti, tsi, wgt, xjac, xn, xnd, xo;

	double schi, si, swgt;
	double result[2];
	double *d, *dt, *dx, *r, *x, *xi, *xin;
	int *ia;

	d = (double*)malloc(sizeof(double) * (NDMX + 1) * (MXDIM + 1)) ;
	dt = (double*)malloc(sizeof(double) * (MXDIM + 1)) ;
	dx = (double*)malloc(sizeof(double) * (MXDIM + 1)) ;  
	r = (double*)malloc(sizeof(double) * (NDMX + 1)) ;
	x = (double*)malloc(sizeof(double) * (MXDIM + 1)) ;
	xi = (double*)malloc(sizeof(double) * (MXDIM + 1) * (NDMX + 1)) ;
	xin = (double*)malloc(sizeof(double) * (NDMX + 1)) ;
	ia = (int*)malloc(sizeof(int) * (MXDIM + 1)) ;

// code works only  for (2 * ng - NDMX) >= 0)

	ndo = 1;
	
	for (j = 1; j <= ndim; j++) 
		xi[j * NDMX1 + 1] = 1.0;
	
	si = swgt = schi = 0.0;
	nd = NDMX;
	ng = 1;
	ng = (int)pow(ncall / 2.0 + 0.25, 1.0 / ndim);
	
	for (k = 1, i = 1; i < ndim; i++) 
		k *= ng;
	
	double sci = 1.0 / k;
	double sc = k;
	k *= ng;
	ncubes = k;
	npg = IMAX(ncall / k, 2);
	calls = (double)npg * (double)k;
	dxg = 1.0 / ng;
	double ing = dxg;
	
	for (dv2g = 1, i = 1; i <= ndim; i++) 
		dv2g *= dxg;
	
	dv2g = (calls * dv2g * calls * dv2g) / npg / npg / (npg - 1.0);
	xnd = nd;
	dxg *= xnd;
	xjac = 1.0 / calls;
	
	for (j = 1; j <= ndim; j++) {
		dx[j] = regn[j + ndim] - regn[j];
		//printf("%e, %e\n", dx[j], xjac);
		xjac *= dx[j];
	}
    
	for (i = 1; i <= IMAX(nd, ndo); i++) 
		r[i] = 1.0;
	
	for (j = 1; j <= ndim; j++) 
		rebin(ndo / xnd, nd, r, xin, &xi[j * NDMX1]);
	ndo = nd;
    
    /*int loopsize = ((NDMX + 1) * (MXDIM + 1));
        for(int i=0; i<loopsize; i++)
            printf("xi[%i]:%.15f\n", i, xi[i]);*/
        
	printf("ng, npg, ncubes, xjac, %d, %d, %12d, %e\n", ng, npg, ncubes, xjac);
	double *d_dev, *dx_dev, *x_dev, *xi_dev, *regn_dev,  *result_dev;
	int *ia_dev;

	cudaMalloc((void**)&result_dev, sizeof(double) * 2); cudaCheckError();  //d_result in new
	cudaMalloc((void**)&d_dev, sizeof(double) * (NDMX + 1) * (MXDIM + 1)); cudaCheckError(); //d_xi in new
	cudaMalloc((void**)&dx_dev, sizeof(double) * (MXDIM + 1)); cudaCheckError(); //d_dx in new
	cudaMalloc((void**)&x_dev, sizeof(double) * (MXDIM + 1)); cudaCheckError();
	cudaMalloc((void**)&xi_dev, sizeof(double) * (MXDIM + 1) * (NDMX + 1)); cudaCheckError(); //d_d in new
	cudaMalloc((void**)&regn_dev, sizeof(double) * ((ndim * 2) + 1)); cudaCheckError(); //d_regn in new
	cudaMalloc((void**)&ia_dev, sizeof(int) * (MXDIM + 1)); cudaCheckError();

	cudaMemcpy( dx_dev, dx, sizeof(double) * (MXDIM + 1), cudaMemcpyHostToDevice) ; cudaCheckError();
	cudaMemcpy( x_dev, x, sizeof(double) * (MXDIM + 1), cudaMemcpyHostToDevice) ; cudaCheckError();
	cudaMemcpy( regn_dev, regn, sizeof(double) * ((ndim * 2) + 1), cudaMemcpyHostToDevice) ; cudaCheckError();

	cudaMemset(ia_dev, 0, sizeof(int) * (MXDIM + 1));

	int chunkSize;

	switch (fcode) {
	case 0:
		chunkSize = 2048;
		break;
	case 1:
		chunkSize = 32;
		break;
	case 2:
		chunkSize = 2048;
		break;
	default:
		chunkSize = 32;
		break;
	}

    
	uint32_t totalNumThreads = (uint32_t) ((ncubes + chunkSize - 1) / chunkSize);
	uint32_t totalCubes = totalNumThreads * chunkSize;
	int extra = totalCubes - ncubes;
	int LastChunk = chunkSize - extra;
	uint32_t nBlocks = ((uint32_t) (((ncubes + BLOCK_DIM_X - 1) / BLOCK_DIM_X)) / chunkSize) + 1;
	uint32_t nThreads = BLOCK_DIM_X;
	printf("ncubes %d nBlocks %d nThreads %d totalNumThreads %d totalCubes %d extra  %d LastChunk %d\n",
	       ncubes, nBlocks, nThreads, totalNumThreads, totalCubes, extra, LastChunk);

	printf("the number of evaluation will be %e\n", calls);
	for (it = 1; it <= itmax; it++) {

		ti = tsi = 0.0;
		for (j = 1; j <= ndim; j++) {
			for (i = 1; i <= nd; i++) 
				d[i * MXDIM1 + j] = 0.0;
		}


		cudaMemcpy( xi_dev, xi, sizeof(double) * (MXDIM + 1) * (NDMX + 1), cudaMemcpyHostToDevice) ; cudaCheckError();
		cudaMemset(d_dev, 0, sizeof(double) * (NDMX + 1) * (MXDIM + 1));
		cudaMemset(result_dev, 0, 2 * sizeof(double));
        printf("First kernel\n");
		vegas_kernel <<< nBlocks, nThreads>>>(ng, ndim, npg, xjac, dxg, result_dev, xnd,
		                                      xi_dev, d_dev, dx_dev, regn_dev, ncubes, it, sc,
		                                      sci,  ing, chunkSize, totalNumThreads,
		                                      LastChunk, fcode);


		cudaMemcpy(xi, xi_dev, sizeof(double) * (MXDIM + 1) * (NDMX + 1), cudaMemcpyDeviceToHost); cudaCheckError();
		cudaMemcpy( d, d_dev,  sizeof(double) * (NDMX + 1) * (MXDIM + 1), cudaMemcpyDeviceToHost) ; cudaCheckError();

		cudaMemcpy(result, result_dev, sizeof(double) * 2, cudaMemcpyDeviceToHost);

		//printf("ti is %f", ti);
		ti  = result[0];
		tsi = result[1];
		tsi *= dv2g;
		printf("iter = %d  integ = %e   std = %e\n", it, ti, sqrt(tsi));

		if (it > skip) {
			wgt = 1.0 / tsi;
			si += wgt * ti;
			schi += wgt * ti * ti;
			swgt += wgt;
			*tgral = si / swgt;
			*chi2a = (schi - si * (*tgral)) / (it - 0.9999);
			if (*chi2a < 0.0) *chi2a = 0.0;
			*sd = sqrt(1.0 / swgt);
			tsi = sqrt(tsi);
			//printf("it %d\n", it);
			printf("%5d   %14.7g+/-%9.2g  %9.2g\n", it, *tgral, *sd, *chi2a);
		}
		//printf("%3d   %e  %e\n", it, ti, tsi);



		for (j = 1; j <= ndim; j++) {
			xo = d[1 * MXDIM1 + j];
			xn = d[2 * MXDIM1 + j];
			d[1 * MXDIM1 + j] = (xo + xn) / 2.0;
			dt[j] = d[1 * MXDIM1 + j];
			for (i = 2; i < nd; i++) {
				rc = xo + xn;
				xo = xn;
				xn = d[(i + 1) * MXDIM1 + j];
				d[i * MXDIM1 + j] = (rc + xn) / 3.0;
				dt[j] += d[i * MXDIM1 + j];
			}
			d[nd * MXDIM1 + j] = (xo + xn) / 2.0;
			dt[j] += d[nd * MXDIM1 + j];
			//printf("iter, j, dtj:    %d    %d      %e\n", it, j, dt[j]);
		}

		for (j = 1; j <= ndim; j++) {
			if (dt[j] > 0.0) {
				rc = 0.0;
				for (i = 1; i <= nd; i++) {
					//if (d[i * MXDIM1 + j] < TINY) d[i * MXDIM1 + j] = TINY;
					r[i] = pow((1.0 - d[i * MXDIM1 + j] / dt[j]) /
					           (log(dt[j]) - log(d[i * MXDIM1 + j])), ALPH);
					rc += r[i];
				}

				rebin(rc / xnd, nd, r, xin, &xi[j * NDMX1]);
			}

		}

	}  // end of iterations

	//  Start of iterations without adjustment

	cudaMemcpy( xi_dev, xi, sizeof(double) * (MXDIM + 1) * (NDMX + 1), cudaMemcpyHostToDevice) ; cudaCheckError();
    int loopsize = ((NDMX + 1) * (MXDIM + 1));
        for(int i=0; i<loopsize; i++)
            printf("sxi[%i]:%.15f\n", i, xi[i]);
	for (it = itmax + 1; it <= titer; it++) {

		ti = tsi = 0.0;

		cudaMemset(result_dev, 0, 2 * sizeof(double));
        printf("Second kernel\n");
		vegas_kernelF <<< nBlocks, nThreads>>>(ng, ndim, npg, xjac, dxg, result_dev, xnd,
		                                       xi_dev, d_dev, dx_dev, regn_dev, ncubes, it, sc,
		                                       sci,  ing, chunkSize, totalNumThreads,
		                                       LastChunk, fcode);


		cudaMemcpy(result, result_dev, sizeof(double) * 2, cudaMemcpyDeviceToHost);

		//printf("ti is %f", ti);
		ti  = result[0];
		tsi = result[1];
		tsi *= dv2g;
		printf("iter = %d  integ = %e   std = %e\n", it, ti, sqrt(tsi));

		wgt = 1.0 / tsi;
		si += wgt * ti;
		schi += wgt * ti * ti;
		swgt += wgt;
		*tgral = si / swgt;
		*chi2a = (schi - si * (*tgral)) / (it - 0.9999);
		if (*chi2a < 0.0) *chi2a = 0.0;
		*sd = sqrt(1.0 / swgt);
		tsi = sqrt(tsi);
		//printf("it %d\n", it);
		printf("%5d   %14.7g+/-%9.4g  %9.2g\n", it, *tgral, *sd, *chi2a);
		//printf("%3d   %e  %e\n", it, ti, tsi);

	}  // end of iterations



	free(d);
	free(dt);
	free(dx);
	free(ia);
	free(x);
	free(xi);

	cudaFree(d_dev);
	cudaFree(dx_dev);
	cudaFree(ia_dev);
	cudaFree(x_dev);
	cudaFree(xi_dev);
	cudaFree(regn_dev);



}

int main(int argc, char **argv)
{

	if (argc < 9) {
		printf( "****************************************\n"
		        "Usage (6 arguments):\n"
		        "./vegas_mcubes FCODE  DIM LL  UL  NCALLS  SKIP\n"
		        "FCODE = 0 to MAX_NUMBER_OF_FUNCTIONS-1\n"
		        "NCALLS in scientific notation, e.g. 1.0E+07 \n"
		        "****************************************\n");
		exit(-1);
	}
    
	int  j;
	double avgi, chi2a, sd;
	double regn[2 * MXDIM + 1];
    
	int fcode = atoi(argv[1]);
	int ndim = atoi(argv[2]);
	float LL = atof(argv[3]);
	float UL = atof(argv[4]);
	double ncall = atof(argv[5]);
	int titer = atoi(argv[6]);
	int itmax = atoi(argv[7]);
	int skip = atoi(argv[8]);

	avgi = sd = chi2a = 0.0;
	for (j = 1; j <= ndim; j++) {
		regn[j] = LL;
		regn[j + ndim] = UL;
	}
    
	vegas(regn, ndim, fcode, ncall, &avgi, &sd, &chi2a, titer, itmax, skip);
	printf("Number of iterations performed: %d\n", itmax);
	printf("Integral, Standard Dev., Chi-sq. = %.18f %.20f% 12.6f\n", avgi, sd, chi2a);
	return 0;

}


