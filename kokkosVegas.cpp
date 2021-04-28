/*

code works for gaussian and sin using switch statement. device pointerr/template slow
down the code by 2x

chunksize needs to be tuned based on the ncalls. For now hardwired using a switch statement


nvcc -O2 -DCUSTOM -o vegas vegas_mcubes.cu -arch=sm_70
OR
nvcc -O2 -DCURAND -o vegas vegas_mcubes.cu -arch=sm_70

example run command

kokkos test ./vegas 0 6 0.0  10.0  2.0E+09  58, 0, 0

nvprof ./vegas 0 6 0.0  10.0  2.0E+09  58, 0, 0

nvprof  ./vegas 1 9 -1.0  1.0  1.0E+07 15 10 10

nvprof ./vegas 2 2 -1.0 1.0  1.0E+09 1 0 0

Last three arguments are: total iterations, iteration

#if defined(KOKKOS_ENABLE_CUDA)
  s << "macro  KOKKOS_ENABLE_CUDA      : defined" << std::endl;
#endif

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>
#include <stdint.h>
#include <ctime>
#include <sys/time.h>
#include <Kokkos_Core.hpp>

#define WARP_SIZE 32
#define BLOCK_DIM_X 128
#define ALPH 1.5
#define NDMX  500
#define MXDIM 20

#define NDMX1 NDMX+1
#define MXDIM1 MXDIM+1

#define PI 3.14159265358979323846
#define CUSTOM
//#define KOKKOS_ENABLE_CUDA_LAMBDA
//#define KOKKOS_ENABLE_CUDA

#include "func.cuh"

typedef Kokkos::View<int*>   ViewVectorInt;
typedef Kokkos::View<float*>   ViewVectorFloat;
typedef Kokkos::View<double*>   ViewVectorDouble;

typedef Kokkos::TeamPolicy<>    team_policy;
typedef Kokkos::TeamPolicy<>::member_type  member_type;

typedef Kokkos::TeamPolicy<>    team_policy;
typedef Kokkos::TeamPolicy<>::member_type  member_type;

typedef Kokkos::
View<double*, Kokkos::DefaultExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > ScratchViewDouble;

//int scratch_size = ScratchViewType::shmem_size(256);

#define IMAX(a,b) \
    ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
      _a > _b ? _a : _b; })

#define IMIN(a,b) \
    ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
      _a < _b ? _a : _b; })

__inline__  __host__ __device__  
void get_indx(int ms, int *da, int ND, int NINTV) {
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

void vegas_kernel_kokkos(int ng, int ndim, int npg, double xjac, double dxg,
                          ViewVectorDouble result_dev, double xnd, ViewVectorDouble xi,
                          ViewVectorDouble d, ViewVectorDouble dx, ViewVectorDouble regn, int ncubes,
                          int iter, double sc, double sci, double ing,
                          int chunkSizeG, uint32_t totalNumThreads,
                          int LastChunk, int fcode){
    
	uint32_t nBlocks = ((uint32_t) (((ncubes + BLOCK_DIM_X - 1) / BLOCK_DIM_X)) / chunkSizeG) + 1;
	uint32_t nThreads = BLOCK_DIM_X;
    //printf("First kernel nBlocks:%i nThreads:%i\n", nBlocks, nThreads);
	//int shMemBytes =  ScratchViewDouble::shmem_size(BLOCK_DIM_X) + ScratchViewDouble::shmem_size(BLOCK_DIM_X);
	Kokkos::parallel_for( "vegas_kernelF",
	                      team_policy(nBlocks, nThreads).set_scratch_size( 0, Kokkos::PerTeam(2048)),
						  KOKKOS_LAMBDA ( const member_type team_member) 
	{
		ScratchViewDouble sh_buff(team_member.team_scratch(0), 256);
		int chunkSize = chunkSizeG;
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
		int tx = team_member.team_rank ();  //local id
		int m = team_member.league_rank () * BLOCK_DIM_X + tx;  //global thread id

		double fb, f2b, wgt, xn, xo, rc, f, f2, ran00;
		int kg[MXDIM + 1];
		int ia[MXDIM + 1];
		double x[MXDIM + 1];
		int k, j;
		double fbg, f2bg;
        
        //if(m == 0)
        //    printf("vegas_kernel_kokkos m:%i totalNumThreads:%i\n", m, totalNumThreads);
        
		if (m < totalNumThreads) {
			if (m == totalNumThreads - 1) 
                chunkSize = LastChunk + 1;
			seed = seed_init + m * chunkSize;
#ifdef CURAND
			curandState localState;
			curand_init(seed, 0, 0, &localState);
#endif
			fbg = f2bg = 0.0;
			get_indx(m * chunkSize, &kg[1], ndim, ng); 
           
			for (int t = 0; t < chunkSize; t++) {                
				fb = f2b = 0.0;

				for ( k = 1; k <= npg; k++) {
					wgt = xjac;
                                        
					for ( int j = 1; j <= ndim; j++) {
#ifdef CUSTOM           
						temp =  a * seed + c;
						seed = temp & (p - 1);
						ran00 = (double) seed / (double) p ;
#endif
#ifdef CURAND
						ran00 = curand_uniform(&localState);
#endif
						xn = (kg[j] - ran00) * dxg + 1.0;
						ia[j] = IMAX(IMIN((int)(xn), NDMX), 1); //this is the bin, different on each dim
						/*if(m == 345)
							printf("iaj:%i dim%i bin:(%.15f, %.15f) chunk:%i npg:%i bin index:%i\n", 
                                ia[j], 
                                j,  
                                xi[j * NDMX1 + ia[j] - 1], 
                                xi[j * NDMX1 + ia[j]], 
                                t, 
                                k,
								j * NDMX1 + ia[j] - 1);*/
                                
						if (ia[j] > 1) {
							xo = xi[j * NDMX1 + ia[j]] - xi[j * NDMX1 + ia[j] - 1]; //bin length?
							rc = xi[j * NDMX1 + ia[j] - 1] + (xn - ia[j]) * xo;
						} else {
							xo = xi[j * NDMX1 + ia[j]];
							rc = (xn - ia[j]) * xo;
						}                     
                        
						x[j] = regn[1] + rc * dx[1]; //assumes regn[i] are all the same or does it have to do with 1D version? in other kernel this is based on j not 1
						wgt *= xo * xnd;
					}
                    
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
                                       
					f = wgt * tmp; 
					f2 = f * f;

					fb += f;
					f2b += f2;
                    
                    for ( j = 1; j <= ndim; j++) {
                        Kokkos::atomic_add(&d[ia[j]*MXDIM1 + j], fabs(f));
                        //d[ia[j]*MXDIM1 + j] += f2;
                    }
                    
				}  // end of npg loop
                                
				f2b = sqrt(f2b * npg); //same
				f2b = (f2b - fb) * (f2b + fb); //same

				fbg += fb;	//same as kernelF
				f2bg += f2b;  //same as kernelF
				
				//same loop as KernelF
				for (int k = ndim; k >= 1; k--) {
					kg[k] %= ng;
					if (++kg[k] != 1) 
                        break;
				}
			} //end of chunk for loop
			
			sh_buff(tx) = fbg;
			sh_buff(tx + BLOCK_DIM_X) = f2bg;
			team_member.team_barrier();
            
			if (tx == 0) {
				double fbgs = 0.0;
				double f2bgs = 0.0;
				for (int ii = 0; ii < BLOCK_DIM_X; ++ii)
				{
					fbgs += sh_buff(ii);
					f2bgs += sh_buff(ii + BLOCK_DIM_X);
				}
				Kokkos::atomic_add(&result_dev[0], fbgs);
				Kokkos::atomic_add(&result_dev[1], f2bgs);               
			}
		} // end of subcube if
	});
}
						  					
void vegas_kernelF_kokkos(uint32_t nBlocks, uint32_t nThreads, int ng, int ndim, int npg, double xjac, double dxg,
                          ViewVectorDouble result_dev, double xnd, ViewVectorDouble xi,
                          ViewVectorDouble d, ViewVectorDouble dx, ViewVectorDouble regn, int ncubes,
                          int iter, double sc, double sci, double ing,
                          int chunkSizeG, uint32_t totalNumThreads,
                          int LastChunk, int fcode) {
    
	Kokkos::parallel_for( "vegas_kernelF",
	                      team_policy(nBlocks, nThreads).set_scratch_size( 0, Kokkos::PerTeam(2048)),
	KOKKOS_LAMBDA ( const member_type team_member) {

    ScratchViewDouble sh_buff(team_member.team_scratch(0), 256);
	
	//	ScratchViewType sh_buff(team_member.team_scratch( 0 ), 256 );

		int chunkSize = chunkSizeG;
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
		int tx = team_member.team_rank ();  //local id
		int m = team_member.league_rank () * BLOCK_DIM_X + tx;  //global thread id

		double fb, f2b, wgt, xn, xo, rc, f, f2, ran00;
		int kg[MXDIM + 1];
		int iaj;
		double x[MXDIM + 1];
		int k;
		double fbg, f2bg;

		if (m < totalNumThreads) {
			if (m == totalNumThreads - 1) 
                chunkSize = LastChunk + 1;
			seed = seed_init + m * chunkSize;
#ifdef CURAND
			curandState localState;
			curand_init(seed, 0, 0, &localState);
#endif
			fbg = f2bg = 0.0;
			get_indx(m * chunkSize, &kg[1], ndim, ng); 

			for (int t = 0; t < chunkSize; t++) {
				fb = f2b = 0.0;

				for ( k = 1; k <= npg; k++) {
					wgt = xjac;

					for ( int j = 1; j <= ndim; j++) {
#ifdef CUSTOM
						temp =  a * seed + c;
						seed = temp & (p - 1);
						ran00 = (double) seed / (double) p ;
#endif
#ifdef CURAND
						ran00 = curand_uniform(&localState);
#endif
                        
                        //dxg is constant across all samples, what does it represent?
                        //what is kg?
						xn = (kg[j] - ran00) * dxg + 1.0;
						iaj = IMAX(IMIN((int)(xn), NDMX), 1); //this is the bin, different on each dim
                        //all dims except 1, read the wrong bin                   
						if (iaj > 1) {
							xo = xi[j * (NDMX + 1) + iaj] - xi[j * (NDMX + 1) + iaj - 1]; //bin length?
							rc = xi[j * (NDMX + 1) + iaj - 1] + (xn - iaj) * xo;
						} else {
							xo = xi[j * (NDMX + 1) + iaj];
							rc = (xn - iaj) * xo;  
						}                     
                                              
						x[j] = regn[1] + rc * dx[1]; //assumes regn[i] are all the same or does it have to do with 1D version? in other kernel this is based on j not 1
						wgt *= xo * xnd;
					}
					//double tmp = func[1](x, ndim);
					double tmp;
                    //printf("calling function fcode:%i\n", fcode);
                    
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

			//	fbg  = blockReduceSum(fbg);
			//	f2bg = blockReduceSum(f2bg);

			sh_buff[tx] = fbg;
			sh_buff[tx + BLOCK_DIM_X] = f2bg;
            
			team_member.team_barrier();

			if (tx == 0) {
				double fbgs = 0.0;
				double f2bgs = 0.0;
				for (int ii = 0; ii < BLOCK_DIM_X; ++ii)
				{
					fbgs += sh_buff[ii];
					f2bgs += sh_buff[ii + BLOCK_DIM_X];
				}
				Kokkos::atomic_add(&result_dev[0], fbgs);
				Kokkos::atomic_add(&result_dev[1], f2bgs);
			}
		} // end of subcube if
	});
}

void rebin(double rc, int nd, double *r, double *xin, ViewVectorDouble::HostMirror xi, int offset)
{
	int i, k = 0;
	double dr = 0.0, xn = 0.0, xo = 0.0;
	for (i = 1; i < nd; i++) {
		while (rc > dr)
			dr += r[++k];
		if (k > 1) 
            xo = xi[offset + k - 1];
		xn = xi[offset + k];
		dr -= rc;
		xin[i] = xn - (xn - xo) * dr / r[k];
	}

	for (i = 1; i < nd; i++){
        //printf("setting x[%i] for bin %i:%.15e\n", i+offset, i,  xin[i]);
        xi[i + offset] = xin[i];
    }
	xi[nd + offset] = 1.0;

}

void vegas(float LL, float UL, int ndim, int fcode,
           double ncall, double * tgral, double * sd,
           double * chi2a, int titer, int itmax, int skip)
{
	Kokkos::initialize();
	{
		int i, it, j, k, nd, ndo, ng, npg, ncubes;
		double calls, dv2g, dxg, rc, ti, tsi, wgt, xjac, xn, xnd, xo;
		double schi, si, swgt;
		
		ViewVectorDouble d_result( "result", 2);    //result_dev in the original
		ViewVectorDouble d_xi( "xi", ((NDMX + 1) * (MXDIM + 1)) );//xi_dev in the original  
		ViewVectorDouble d_d( "d", ((NDMX + 1) * (MXDIM + 1)) );   //d_dev in the original
		ViewVectorDouble d_dx( "dx", MXDIM + 1 );   //dx_dev in the original
		ViewVectorDouble d_regn( "regn", 2 * MXDIM + 1); //regn_dev in the original
        
        printf("d_xi size:%lu\n", ((NDMX + 1) * (MXDIM + 1)));
        printf("d_d size:%lu\n", ((NDMX + 1) * (MXDIM + 1)));
        printf("dx size:%lu\n", MXDIM + 1 );
        printf("d_regn size:%lu\n", 2 * MXDIM + 1);
            
        // create host mirrors of device views
		ViewVectorDouble::HostMirror result = Kokkos::create_mirror_view(d_result);
		ViewVectorDouble::HostMirror xi     = Kokkos::create_mirror_view(d_xi); //left coordinate of bin
		ViewVectorDouble::HostMirror d      = Kokkos::create_mirror_view(d_d);
		ViewVectorDouble::HostMirror dx     = Kokkos::create_mirror_view(d_dx);
		ViewVectorDouble::HostMirror regn   = Kokkos::create_mirror_view(d_regn);
        
		for (j = 1; j <= ndim; j++) {
			regn[j] = LL;
			regn[j + ndim] = UL;
		}

        // create arrays used only on host
		double *dt, *r, *xin;
		dt =  (double*)malloc(sizeof(double) * (MXDIM + 1)) ;
		r =   (double*)malloc(sizeof(double) * (NDMX + 1)) ;
		xin = (double*)malloc(sizeof(double) * (NDMX + 1)) ;

        // code works only  for (2 * ng - NDMX) >= 0)
		ndo = 1;
		for (j = 1; j <= ndim; j++) 
            xi(j * (NDMX + 1) + 1) = 1.0;
            
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
        printf("ncubes=k:%i (ng^NDIM)\n", ncubes);
        
		npg = IMAX(ncall / k, 2);
        printf("npg:%i (ncall/k ^2)\n", npg);
        
		calls = (double)npg * (double)k;
		dxg = 1.0 / ng;
		double ing = dxg;
        
		for (dv2g = 1, i = 1; i <= ndim; i++) 
            dv2g *= dxg;
            
		dv2g = (calls * dv2g * calls * dv2g) / npg / npg / (npg - 1.0);
		xnd = nd;
		dxg *= xnd;
		xjac = 1.0 / calls;
        printf("nd:%i ng:%i dxg:%.15e\n", nd, ng, dxg);
		for (j = 1; j <= ndim; j++) {
			dx(j) = regn[j + ndim] - regn[j];
			printf("dx[j]:%e, xjac:%e\n", dx(j), xjac);
			xjac *= dx(j);
		}

		for (i = 1; i <= IMAX(nd, ndo); i++) 
            r[i] = 1.0;
        
        printf("ndo:%i xnd:%.15e ndo/xnd:%.15e is the size of each bin\n", ndo, xnd, ndo/xnd);
		for (j = 1; j <= ndim; j++){ 
            printf("rebin offset:%i\n", j * (NDMX + 1));
            rebin(ndo / xnd, nd, r, xin, xi, j * (NDMX + 1));
        }
            
		ndo = nd;
        
        
        
		printf("ng, npg, ncubes, xjac, %d, %d, %12d, %e\n", ng, npg, ncubes, xjac);
        
		Kokkos::deep_copy( d_dx, dx);
		Kokkos::deep_copy( d_regn, regn);
        
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

        printf("itmax:%i titer:%i\n", itmax, titer);
        //--------------------------------------------------------------------------------------------------
        for (it = 1; it <= itmax; it++) {

            ti = tsi = 0.0;
            for (j = 1; j <= ndim; j++) {
                for (i = 1; i <= nd; i++) 
                    d(i * MXDIM1 + j) = 0.0;
            }

            deep_copy( d_xi, xi) ;
            //cudaMemset(d_dev, 0, sizeof(double) * (NDMX + 1) * (MXDIM + 1));
            //cudaMemset(result_dev, 0, 2 * sizeof(double));
            printf("First kernel\n");
            vegas_kernel_kokkos(ng, ndim, npg, xjac, dxg, d_result, xnd,
		                                      d_xi, d_d, d_dx, d_regn, ncubes, it, sc,
		                                      sci,  ing, chunkSize, totalNumThreads,
		                                      LastChunk, fcode);
            printf("Execution of vegas_kernel_kokkos at iteration %i done\n", it);
            deep_copy(xi, d_xi);
            //cudaMemcpy(xi, xi_dev, sizeof(double) * (MXDIM + 1) * (NDMX + 1), cudaMemcpyDeviceToHost); cudaCheckError();
            deep_copy(d, d_d);
            //cudaMemcpy( d, d_dev,  sizeof(double) * (NDMX + 1) * (MXDIM + 1), cudaMemcpyDeviceToHost) ; cudaCheckError();
            deep_copy(result, d_result);
            //cudaMemcpy(result, result_dev, sizeof(double) * 2, cudaMemcpyDeviceToHost);

            ti  = result(0);
            tsi = result(1);
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
                printf("%5d   %14.7g+/-%9.2g  %9.2g\n", it, *tgral, *sd, *chi2a);
            }

            for (j = 1; j <= ndim; j++) {
                xo = d(1 * MXDIM1 + j);
                xn = d(2 * MXDIM1 + j);
                d(1 * MXDIM1 + j) = (xo + xn) / 2.0;
                dt[j] = d(1 * MXDIM1 + j);
                for (i = 2; i < nd; i++) {
                    rc = xo + xn;
                    xo = xn;
                    xn = d[(i + 1) * MXDIM1 + j];
                    d(i * MXDIM1 + j) = (rc + xn) / 3.0;
                    dt[j] += d(i * MXDIM1 + j);
                }
                d(nd * MXDIM1 + j) = (xo + xn) / 2.0;
                dt[j] += d(nd * MXDIM1 + j);
                //printf("iter, j, dtj:    %d    %d      %e\n", it, j, dt[j]);
            }

            for (j = 1; j <= ndim; j++) {
                if (dt[j] > 0.0) {
                    rc = 0.0;
                    for (i = 1; i <= nd; i++) {
                        r[i] = pow((1.0 - d(i * MXDIM1 + j) / dt[j]) /
					           (log(dt[j]) - log(d(i * MXDIM1 + j))), ALPH);
                        rc += r[i];
                    }

                    rebin(rc / xnd, nd, r, xin, xi, j * NDMX1);//commented out the + 1
                }
            }
        }
        //----------------------------------------------------------------------
            
		printf("the number of evaluation will be %e\n", calls);
		//  Start of iterations without adjustment
        //------------------------------------
        //int loopsize = ((NDMX + 1) * (MXDIM + 1));
        //for(int i=0; i<loopsize; i++)
        //    printf("sxi[%i]:%.15f\n", i, xi[i]);
        //------------------------------------
		Kokkos::deep_copy( d_xi, xi);
		for (it = itmax + 1; it <= titer; it++) {

			ti = tsi = 0.0;
			Kokkos::deep_copy( d_result, 0.0);
            printf("Second kernel\n");
			vegas_kernelF_kokkos(nBlocks, nThreads, ng, ndim, npg, xjac, dxg, d_result, xnd,
			                     d_xi, d_d, d_dx, d_regn, ncubes, it, sc,
			                     sci,  ing, chunkSize, totalNumThreads,
			                     LastChunk, fcode);

			Kokkos::deep_copy( result, d_result);

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
            
			if (*chi2a < 0.0) 
                *chi2a = 0.0;
                
			*sd = sqrt(1.0 / swgt);
			tsi = sqrt(tsi);
			//printf("it %d\n", it);
			printf("%5d   %14.7g+/-%9.4g  %9.2g\n", it, *tgral, *sd, *chi2a);
			//printf("%3d   %e  %e\n", it, ti, tsi);

		}  // end of iterations

		//printf("End of routine\n");

		free(dt);
		free(r);
		free(xin);
	}

	Kokkos::finalize();
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
    //double regn[2 * MXDIM + 1];

	int fcode = atoi(argv[1]);
	int ndim = atoi(argv[2]);
	float LL = atof(argv[3]);
	float UL = atof(argv[4]);
	double ncall = atof(argv[5]);
	int titer = atoi(argv[6]);
	int itmax = atoi(argv[7]);
	int skip = atoi(argv[8]);

	avgi = sd = chi2a = 0.0;
	// for (j = 1; j <= ndim; j++) {
	// 	regn[j] = LL;
	// 	regn[j + ndim] = UL;
	// }


	vegas(LL, UL, ndim, fcode, ncall, &avgi, &sd, &chi2a, titer, itmax, skip);

	printf("Number of iterations performed: %d\n", itmax);

	printf("Integral, Standard Dev., Chi-sq. = %.18f %.20f% 12.6f\n",

	       avgi, sd, chi2a);


	return 0;

}


