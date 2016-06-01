#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <vector>
#include <omp.h>
#include <ctime>
#include <RcppGSL.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_integration.h>
#include <iomanip>      // std::setprecision

// [[Rcpp::plugins(openmp)]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppGSL)]]

#define TARGACCEPT 0.44
#define NORMDIST 1
#define STUDDIST 2
#define HSSTDIST 3

using namespace arma ;
using namespace Rcpp ;


/*  
 * adaptive in the metropolish hasting algorithm
 * @param    iteration
 * @return   min( 0.01, 1.0 / sqrt(iteration) )
 */

double adaptamount( int iteration )
{
    /* return( 1.0 / imax(100, sqrt(iteration)) ); */
    return( std::min( 0.01, 1.0 / sqrt((double)iteration) ) );
    /* return(0.01); */
}


/*  
 * Calculate s_{it} = \nabla_{it} = d( log p(u | z) )/ df
 * @param    Gaussian x,z, rho
 * @return   void s_{it}
 */

// [[Rcpp::export]]
void  dlogc(arma::mat&  x, arma::vec& z,
                 arma::mat& rho, arma::mat& f,
                 arma::mat& psi, int t, int n_max,int pos){
    psi(t,pos) = (x(t,pos) * z(t) /2 + rho(t,pos)/2 - rho(t,pos) /2 * (pow(x(t,pos),2) + pow(z(t),2) - 2 * rho(t,pos) * x(t,pos) * z(t)) / (1-pow(rho(t,pos),2) )) ;
    
}

/*  
 * Calculate s_{t} = \nabla_{t} = d( log p(u | z) )/ df
 * Calculates a vector of psi at time t
 * @param    Gaussian x,z, rho
 * @return   void s_{t}
 */

// [[Rcpp::export]]
void  dlogmvn(arma::mat&  x, arma::vec& z,
                   arma::mat& rho, arma::mat& f,
                   arma::mat& psi, int t, int n_max){
    
    for (int i = 0; i < n_max; ++i){
        psi(t,i) = (x(t,i) * z(t) /2 + rho(t,i)/2 - rho(t,i) /2 * (pow(x(t,i),2) + pow(z(t),2) - 2 * rho(t,i) * x(t,i) * z(t)) / (1-pow(rho(t,i),2) )) ;
    }
    
}


/*  
 * Density for the hyperbolic skew t distribution with df degrees of freedom
 * @param    x, nu , gamma
 * @return  log p(x | nu, gamma)
 */

// [[Rcpp::export]]
arma::colvec dskewt(arma::colvec  x, double nu, double gamma){
    
    int t_max = x.n_rows;
    arma::vec log_dskewt(t_max);
    //change to the standard notation of GH skew-t (mu,delta,beta,nu)
    double mu = 0;
    double delta = sqrt(nu);
    double beta = gamma;
    
        for (int j =0;j < t_max;j++){
            log_dskewt(j) = ((1 - nu)/2) * log(2) + nu * log(delta) +
                ((nu + 1)/2) * log(fabs(beta)) +
                log(gsl_sf_bessel_Knu_scaled((nu + 1)/2,sqrt(pow(beta,2)*(pow(delta,2) + pow(x(j) - mu,2))) )) -
                                    sqrt(pow(beta,2) * (pow(delta,2) + pow(x(j) - mu,2))) +
                                    beta * (x(j) - mu) - lgamma(nu/2) - log(M_PI)/2 -
                                    ((nu + 1)/2) * log(pow(delta,2) + pow(x(j) - mu,2))/2;
        }

    return log_dskewt;
}


struct dskewhyp_params {double mu; double delta;double beta;double nu;}; 

// [[Rcpp::export]]
double dskewhypGSL(double  x_student, void * p){
    struct dskewhyp_params * params = (struct dskewhyp_params *)p;
    
    double mu = (params->mu);
    double delta = (params->delta);
    double beta = (params->beta);
    double nu = (params->nu);
    
    double dskewhyp = ((1 - nu)/2) * log(2) + nu * log(delta) +
        ((nu + 1)/2) * log(fabs(beta)) +
        log(gsl_sf_bessel_Knu_scaled((nu + 1)/2,sqrt(pow(beta,2)*(pow(delta,2) + pow(x_student - mu,2))) )) -
                            sqrt(pow(beta,2) * (pow(delta,2) + pow(x_student - mu,2))) +
                            beta * (x_student - mu) - lgamma(nu/2) - log(M_PI)/2 -
                            ((nu + 1)/2) * log(pow(delta,2) + pow(x_student - mu,2))/2;
    
    return(exp(dskewhyp));
}


/*  
 * Quantile function for the hyperbolic skew t distribution with df degrees of freedom
 * @param    u, nu , gamma
 * @return  x = F^{-1}(u | nu, gamma)
 */

// [[Rcpp::export]]
void qskewt(arma::mat& x_student, arma::mat& u,
            std::vector<int>& gid,std::vector<int>& g_mem,
            double nu,double gamma, 
            int t_max, int n_max, int k, int num_mem){
    
    struct dskewhyp_params params = {0,sqrt(nu),gamma,nu};               
    gsl_integration_workspace * w = gsl_integration_workspace_alloc(1000);
//     Rcpp::NumericVector result(20); 
//     Rcpp::NumericVector error(20); 
    double result; double error;
    
    gsl_function F;
    F.function = &dskewhypGSL;
    F.params = &params;
    
    arma::vec xseq(1033);
    arma::vec Fseq(1033);  
    arma::vec fseq(1033);  
    
    xseq(0) = -27;
    for (int i = 0; i < 18; i++){ xseq(i+1) = xseq(i) +1;}
    for (int i = 18; i < 26; i++){ xseq(i+1) = xseq(i) +0.5;}
    for (int i = 26; i < 1027; i++){ xseq(i+1) = xseq(i) +0.01;}
    for (int i = 1027; i < 1033; i++){ xseq(i) = xseq(i-1) +0.5;}
    
    
    gsl_integration_qagil (&F, xseq(0), 0, 1e-7, 1000,
                           w, &result, &error);
    fseq(0) = result;
    
    for (int i = 1; i < 1033; i++){ 
        gsl_integration_qag (&F, xseq(i-1),xseq(i), 0, 1e-7, 1000,1,
                             w, &result, &error);
        fseq(i) = result;}  
    Fseq = arma::cumsum(fseq);
    
    
    gsl_integration_workspace_free (w);

    arma::vec qskewt_linear(t_max);
    
    #pragma omp parallel for default(none) private(qskewt_linear) shared(Fseq, xseq, u,num_mem,g_mem,x_student)
    for( int g = 0; g < num_mem; g++){
        interp1(Fseq, xseq, u.col(g_mem[g]), qskewt_linear);  
        x_student.col(g_mem[g]) = qskewt_linear;
    }
  
    
    
}

arma::rowvec f0toblock(arma::rowvec& f0, std::vector<int>& gid, int t_max, int n_max, int n_group){
    arma::rowvec frow(n_max);
    for (int i = 0; i < n_max;i++)
    {
        frow(i) = f0(gid[i]);
    }
    return(frow);
}
    

/*  
 * Generate the \rho_{it} processes for Gaussian x
 * @param    u, theta,
 * @return void matrix \rho
 */

// [[Rcpp::export]]
void gen_rho_Gauss(arma::mat& x_normal, 
             arma::vec& z, arma::mat& rho, arma::mat& f,
             arma::mat& psi,std::vector<int>& gid,
             arma::rowvec& a, arma::rowvec& b, arma::rowvec& f0, 
             int t_max, int n_max, int n_group){

    arma::rowvec a_expand(n_max); arma::rowvec b_expand(n_max);
    for( int i = 0; i < n_max; i++){
        f(0,i) =  f0[gid[i]];
        rho(0,i) = (1-exp(-f0[gid[i]]))/(1+exp(-f0[gid[i]]));
        a_expand[i] = a[gid[i]];
        b_expand[i] = b[gid[i]];
    }
    for( int j = 1; j < t_max; j++){
        dlogmvn(x_normal,z,rho,f,psi, j-1, n_max);
        f.row(j) = (1-b_expand) % f.row(0) + a_expand % psi.row(j-1) + b_expand % f.row(j-1);
        for( int pos = 0; pos < n_max; pos++){
            if (fabs( f(j-1,pos)) == 5.5 ){
                if (fabs( f(j,pos)) > 5.5 ){
                    f(j,pos) = f(j-1,pos);
                }
            }   
            else if (fabs( f(j,pos)) > 5.5 ){
                if ( f(j,pos) > 5.5 ){
                    f(j,pos) = 5.5;
                } else  f(j,pos) = -5.5;    
            }
            // if (f(j,pos) < -5.5 || f(j-1,pos) == -5.5) {f(j,pos) = -5.5;}             //adding checking step
            // if (f(j,pos) > 5.5 || f(j-1,pos) == 5.5) {f(j,pos) = 5.5;}             //adding checking step
        }
        rho.row(j) = (1-exp(-f.row(j)))/(1+exp(-f.row(j)));
    }
}

/*  Note: This function for 3 types of distribution
 * Generate the \rho_{it} processes 
 * @param    u, theta,
 * @return void matrix \rho
 */

// [[Rcpp::export]]
void gen_rho(arma::mat& x_student, arma::mat& x_normal, arma::mat& u,
             arma::vec& z, arma::mat& rho, arma::mat& f,
             arma::mat& psi,std::vector<int>& gid,
             std::vector<std::vector<int> >&  g_mat, std::vector<int>& g_count,
             arma::rowvec& a, arma::rowvec& b, arma::rowvec& f0, 
             arma::rowvec& nu,arma::rowvec& gamma, arma::mat& zeta,arma::mat& zeta_sqrt,
             int t_max, int n_max, int n_group, int disttype, bool changeNu){
    // Convert u to x_student with nu degree of freedom
    if (changeNu){
        if (disttype == HSSTDIST){
            for( int k = 0; k < n_group; k++){
                qskewt(x_student, u, gid, g_mat[k], nu[k], gamma[k], t_max,n_max,k,g_count[k] );
            }
            
            for( int i = 0; i < n_max; i++){
                x_normal.col(i) = (x_student.col(i) - zeta.col(gid[i]) * gamma(gid[i]) )/zeta_sqrt.col(gid[i]);
            }
        }
        if (disttype == STUDDIST){
            
            for( int i = 0; i < n_max; i++){
                for( int j = 0; j < t_max; j++){
                    x_student(j,i) = R::qt(u(j,i), nu[gid[i]],1,0);
                }
                x_normal.col(i) = x_student.col(i)/zeta_sqrt.col(gid[i]);
            }

        }
    }
    gen_rho_Gauss(x_normal,z,rho,f,psi,gid,a,b,f0,t_max,n_max,n_group);
}

/* 
 * Generate the \rho_{it} processes for 1 series pos
 * @param    x_norm, theta,
 * @return void vector \rho_{,pos}
 */

// [[Rcpp::export]]
void  gen_rho_paral(arma::mat&  x, arma::vec& z,
                    arma::mat& rho, arma::mat& f,
                    arma::mat& psi,std::vector<int>& gid,
                    arma::rowvec& a, arma::rowvec& b, arma::rowvec f0,
                    int t_max, int n_max,int pos){
    f(0,pos) =  f0(gid[pos]);
    rho(0,pos) = (1-exp(-f0(gid[pos])))/(1+exp(-f0(gid[pos])));
    
    for( int j = 1; j < t_max; j++){
        dlogc(x,z,rho,f,psi, j-1, n_max,pos);
        f(j,pos) = (1-b(gid[pos]))*f0(gid[pos]) + a(gid[pos]) * psi(j-1,pos) + b(gid[pos]) * f(j-1,pos);
        if (fabs( f(j-1,pos)) == 5.5 ){
            if (fabs( f(j,pos)) > 5.5 ){
                f(j,pos) = f(j-1,pos);
            }
        }   
        else if (fabs( f(j,pos)) > 5.5 ){
            if ( f(j,pos) > 5.5 ){
                f(j,pos) = 5.5;
            } else  f(j,pos) = -5.5;    
        }
        // if (f(j,pos) < -5.5 || f(j-1,pos) == -5.5) {f(j,pos) = -5.5;}             //adding checking step
        // if (f(j,pos) > 5.5 || f(j-1,pos) == 5.5) {f(j,pos) = 5.5;}             //adding checking step
        rho(j,pos) = (1-exp(-f(j,pos) ))/(1+exp(-f(j,pos) ));
    }
    
}

/*  Note: This function for 3 types of distribution
 * Generate the \rho_{it} processes in group
 * @param    u, theta,
 * @return void matrix \rho
 */

// [[Rcpp::export]]
void  gen_rho_paralG(arma::mat& x_student, arma::mat& x_normal, arma::mat& u,
                     arma::vec& z, arma::mat& rho, arma::mat& f,
                     arma::mat& psi,std::vector<int>& gid,std::vector<int>& g_mem,
                     arma::rowvec& a, arma::rowvec& b, arma::rowvec f0,
                     arma::rowvec& nu, arma::rowvec& gamma,
                     arma::mat& zeta, arma::mat& zeta_sqrt,
                     int t_max, int n_max,int k, int num_mem,int disttype,  bool changeNu){
    
    // Convert u to x_student with nu degree of freedom
    if (changeNu){
        if (disttype == HSSTDIST){
            qskewt(x_student, u, gid, g_mem, nu[k], gamma[k], t_max,n_max,k,num_mem );
            for( int g = 0; g < num_mem; g++){
                x_normal.col(g_mem[g]) = (x_student.col(g_mem[g]) - zeta.col(k) * gamma(k) )/zeta_sqrt.col(k);
            }
        }
        if (disttype == STUDDIST){
            
            for( int g = 0; g < num_mem; g++){
                for( int j = 0; j < t_max; j++){
                    //std::transform(u.begin(), u.end(), x.begin(), qt_nu);
                    x_student(j,g_mem[g]) = R::qt(u(j,g_mem[g]), nu[k],1,0);
                }
                x_normal.col(g_mem[g]) = x_student.col(g_mem[g])/zeta_sqrt.col(k);
            }
            
        }
        
    }
    

    for( int g = 0; g < num_mem; g++){
        gen_rho_paral(x_normal,z,rho,f,psi,gid,a,b,f0,t_max,n_max,g_mem[g]);
        }
}

/*  
 * Calculate log phi(x_{kt} | z_t*rho_{kt}, sigma_{kt}) for col k
 * @param    x_norm, theta,
 * @return sum_log_phi
 */

// [[Rcpp::export]]
double likelihood_uniseries(arma::mat& x, arma::vec& z, arma::mat& rho, int t_max, int n_max, int k,
                            arma::mat& LL_cond, arma::rowvec& sum_LL_cond,  bool booSum = true){
    arma::colvec Sigmasq = (1 - pow(rho.col(k),2));
    LL_cond.col(k) = - 0.5* ( log(2 * M_PI) + log(Sigmasq) + pow(x.col(k)-rho.col(k) % z,2) / Sigmasq  ) ;
    sum_LL_cond[k] = arma::accu(LL_cond.col(k) );
    if (booSum)
        return (sum_LL_cond[k]);
    else 
        return (0);
}

/*  
 * Calculate log phi(x_{kt} | z_t*rho_{kt}, sigma_{kt}) for group k
 * @param    x_norm, theta,
 * @return sum_log_phi
 */

// [[Rcpp::export]]
double likelihood_group(arma::mat& x, arma::vec& z, arma::mat& rho,
                        std::vector<int>& gid,std::vector<int>& g_mem,
                        int t_max, int n_max, int k, int num_mem, 
                        arma::mat& LL_cond, arma::rowvec& sum_LL_cond,  bool reCal = true){
    double LL_group = 0;
    if (reCal){
        //   #ifdef _OPENMP
        //     omp_set_num_threads(8);
        //   #endif
        //     
        //   #pragma omp parallel for default(none) shared(t_max,n_max,x,z,rho,g_mem,num_mem,LL_ele)
        
        for (int g =0; g < num_mem; g++){
            sum_LL_cond[g_mem[g]] = likelihood_uniseries(x, z, rho, t_max,n_max, g_mem[g], LL_cond, sum_LL_cond, true);
            LL_group += sum_LL_cond[g_mem[g]];
        }
        
        return (LL_group);
        
    }
    else{
        for (int g =0; g < num_mem; g++){
            LL_group += sum_LL_cond[g_mem[g]];
        }
        
        return (LL_group);
        
    }
    
}

/*  
 * Calculate log phi(x_{kt} | z_t*rho_{kt}, sigma_{kt}) for group k at time j
 * @param    x_norm, theta,
 * @return sum_log_phi
 */

// [[Rcpp::export]]
double likelihood_group_t(arma::mat& x_student, arma::mat& x_normal, arma::vec& z, 
                          arma::mat& rho, std::vector<int>& gid,std::vector<int>& g_mem,
                          arma::rowvec& gamma,
                          arma::mat& zeta,arma::mat& zeta_sqrt,
                          int t_max, int n_max, int k, int j, int num_mem, 
                          arma::mat& LL_cond, bool reCal = true){
    double LL_group = 0;
    zeta_sqrt(j,k) = sqrt(zeta(j,k));
    
    for (int g =0; g < num_mem; g++){
        x_normal(j,g_mem[g]) = (x_student(j,g_mem[g]) - zeta(j,k) * gamma(k) )/zeta_sqrt(j,k);
        LL_cond(j,g_mem[g]) = - 0.5* ( log(2 * M_PI) + log(1 - pow(rho(j,g_mem[g]),2)) + pow(x_normal(j,g_mem[g])-rho(j,g_mem[g]) * z(j),2) / (1 - pow(rho(j,g_mem[g]),2))  );
        LL_group += LL_cond(j,g_mem[g]);
    }
    
    return (LL_group);
}

/*  
 * Calculate Score Hessian - based on Creal (2015) function matlab
 * @param    dLog_zeta, theta,
 * @return dHessian
 */

// [[Rcpp::export]]
void calculateScoreHessian(double dLog_zeta,double dAlpha1,double dAlpha2,double dAlpha3,double& dScore,double& dHessian, bool print_out = false){
    double zeta = exp(dLog_zeta);
    //dScore = (-dAlpha1/zeta + dAlpha2/(zeta*zeta) -0.5*dAlpha3*(zeta^(-3/2)) ) * zeta;
    //dHessian = (dAlpha1/(zeta*zeta) -2*dAlpha2/(zeta^3) + (3/4)*dAlpha3*(zeta^(-5/2)) ) * zeta^2;
    dScore = -dAlpha1 + dAlpha2/zeta -0.5*dAlpha3*(pow(zeta,-0.5)) ;
    dHessian = dAlpha1 -2*dAlpha2/zeta + (0.75)*dAlpha3*(pow(zeta,-0.5)) ;
    // if (print_out) {std::cout << dScore << " " << dHessian << std::endl;}
}

/*  
 * Find_target mean and variance for zeta - based on Creal (2015) function matlab
 * @param    x_student, theta,
 * @return dHessian
 */

// [[Rcpp::export]]
void find_target(arma::mat& x_student, arma::mat& x_normal, arma::vec& z, 
                 arma::mat& rho, std::vector<int> gid,std::vector<int> g_mem,
                 arma::mat& zeta,arma::mat& zeta_sqrt, double nu,
                 int t_max, int n_max, int k, int j, int num_mem, double& mode_target, double& cov_target){
    
    double alpha1 = 0;
    double alpha2 = 0;
    double alpha3 = 0;
    alpha1 = (nu + num_mem)/2 + 1;
    alpha2 = nu;
    
    for( int g = 0; g < num_mem; g++){
        alpha2 += x_student(j,g_mem[g]) * x_student(j,g_mem[g]) / (1 - pow(rho(j,g_mem[g]),2));
        alpha3 += x_student(j,g_mem[g]) * rho(j,g_mem[g])*z(j)  / (1 - pow(rho(j,g_mem[g]),2));
    }
    alpha2 = alpha2/2;
    int iter = 0;
    double tolerance = 0.000001;
    //        double eval = log(zeta(j,k)); // initial guess is the same as the old value
    double eval = log(0.15); // initial guess is the same as the old value
    double dist = 10;
    double dScore = 0;
    double dHessian = 0;
    iter = 0;
    do{
        calculateScoreHessian(eval, alpha1, alpha2, alpha3, dScore, dHessian);
        dist = - dScore/dHessian;
        iter += 1;
        eval = eval + dist;
    }while ((fabs(dist)>=tolerance)&&(iter<50));
    mode_target = eval;
    cov_target = -1/dHessian;
}

/*  
 * Calculate log scale student density with mean and sigma2
 * @param    y, mu,sigma2, nu
 * @return log f(y)
 */
double logStudentDensity(double y, double mu, double sigma2, double nu){
    return(   R::lgammafn((nu + 1)/2) - R::lgammafn(nu/2) -0.5*log(nu*M_PI) -0.5*log(sigma2)-((nu+1)/2)*log(1 + (pow(y-mu,2))/(nu*sigma2)));
}



/*  
 * Calculate log phi(x_{kt} | z_t*rho_{kt}, sigma_{kt}) for all series
 * @param    x_norm, theta,
 * @return sum_log_phi
 */

// [[Rcpp::export]]
double likelihood(arma::mat& x, arma::vec& z, arma::mat& rho, int t_max, int n_max,
                  arma::mat& LL_cond, arma::rowvec& sum_LL_cond,  bool booSum = true){
    for (int g =0; g < n_max; g++){
        sum_LL_cond[g] = likelihood_uniseries(x, z, rho, t_max,n_max, g, LL_cond, sum_LL_cond, true);
        // Rcpp::Rcout << "g " << g << " " << LL_ele[g] << std::endl;
    }
    if (booSum)
        return (arma::accu(sum_LL_cond));
    else 
        return (0);
}


/*  
 * Sum Density for the hyperbolic skew t distribution with df degrees of freedom
 * @param    x, nu , gamma
 * @return  sum log p(x | nu, gamma)
 */

// [[Rcpp::export]]
double log_studentG(arma::mat& x_student, std::vector<int>& gid, std::vector<int>& g_mem, double nu, double gamma, int t_max, int n_max,int k, int num_mem, int disttype){
    double LL_student = 0;
    if (disttype == HSSTDIST){
        arma::colvec fseq(x_student.n_rows);
        for (int g =0; g < num_mem; g++){
            fseq = dskewt(x_student.col(g_mem[g]),nu,gamma);
            LL_student += sum(fseq);
        }
        
    } else {
        for (int t =0; t < t_max; t++){
            for (int g =0; g < num_mem; g++){
                LL_student += R::dt(x_student(t,g_mem[g]),nu, true);
            }
        }  
    }
    return (LL_student);
}



/*  
 * Sum Density for the InvGamma(zeta |nu/2, nu/2)
 * @param    zeta, nu/2 
 * @return  log p(zeta | nu/2, nu/2)
 */

// [[Rcpp::export]]
double log_InvGammaG(arma::mat& zeta, double nu_div2, int t_max, int n_max,int k){
    double LL_InvGammaG = 0;
    for (int i =0; i < t_max; i++){
        LL_InvGammaG += -log(zeta(i,k))*(nu_div2+1) - nu_div2/zeta(i,k);
    }
    LL_InvGammaG += t_max * nu_div2 * log(nu_div2) - t_max * R::lgammafn(nu_div2); 	
    return (LL_InvGammaG);
}


void model_select_LL(arma::mat& x_student, arma::mat& x_normal,
                     arma::vec& z, std::vector<int>& gid,
                     std::vector<std::vector<int> >&  g_mat, std::vector<int>& g_count,
                     arma::rowvec& nu,arma::rowvec& gamma, arma::mat& zeta, 
                     arma::rowvec& sum_LL_cond,
                     arma::rowvec& LL_rec, arma::rowvec& LL_rec_jwZ,
                     int t_max, int n_max, int n_group, int i, int disttype){
    
    arma::rowvec sum_LL_stu(n_group, fill::zeros);
    arma::rowvec sum_LL_gamma(n_group, fill::zeros);    
    
    LL_rec[i] = accu(sum_LL_cond);
    //Rcpp::Rcout << "i " << i << " LL_rec " << LL_rec(i) << " x_normal " << - arma::accu(-0.5*(log(2 * M_PI) + pow(x_normal,2) ) ) << std::endl;
    
    if (disttype == NORMDIST){
        LL_rec[i] = LL_rec[i] - arma::accu(-0.5*(log(2 * M_PI) + pow(x_normal,2) ) );
        //Rcpp::Rcout << "i " << i << " LL_rec " << LL_rec(i) << " x_normal " << - arma::accu(-0.5*(log(2 * M_PI) + pow(x_normal,2) ) ) << std::endl;
        
    } else{
        #pragma omp parallel for default(none) shared(x_student,z,gid,g_mat,nu,gamma,zeta, t_max,n_max,g_count,n_group,sum_LL_stu,sum_LL_gamma, disttype)                
        for (int k = 0; k < n_group; k++){
            sum_LL_stu[k] = - log_studentG(x_student, gid, g_mat[k],nu[k],gamma[k],t_max, n_max, k,g_count[k], disttype);
            sum_LL_gamma[k] = - (g_count[k] + 1)*0.5*arma::accu(log(zeta.col(k)));
        }
        //Rcpp::Rcout << "i " << i << " LL_rec " << LL_rec(i) << " x_normal " << - arma::accu(-0.5*(log(2 * M_PI) + pow(x_normal,2) ) ) << std::endl;
        //Rcpp::Rcout << "i " << i << " LL_rec " << LL_rec(i) << " x_student " <<  accu(sum_LL_marginal) << std::endl;
        
        LL_rec[i] = LL_rec[i] +  accu(sum_LL_stu) + accu(sum_LL_gamma) ;
        //Rcpp::Rcout << "i " << i << " sum_LL_stu " <<  accu(sum_LL_stu) << " sum_LL_gamma " <<  accu(sum_LL_gamma) << std::endl;
    }
    
    
    LL_rec_jwZ[i] = LL_rec[i] + arma::accu(-0.5*(log(2 * M_PI) + pow(z,2) ) );
    if (disttype > NORMDIST){
        #pragma omp parallel for default(none) shared(x_student,z,gid,g_mat,nu,gamma,zeta, t_max,n_max,g_count,n_group,sum_LL_cond,sum_LL_gamma, disttype)                
        for (int k = 0; k < n_group; k++){
            sum_LL_gamma[k] = log_InvGammaG(zeta,nu(k)/2, t_max, n_max, k);
        }
        LL_rec_jwZ[i] = LL_rec_jwZ[i] +  accu(sum_LL_gamma);    
        
    }
    
}




// [[Rcpp::export]]
List MCMCstep(SEXP data_, SEXP init_,SEXP iter_,SEXP numbatches_,SEXP batchlength_, SEXP other_) {
    BEGIN_RCPP
    
    int seed = 126;
    std::srand(seed);
    
    // Data input
    Rcpp::List data(data_);
    int t_max  = as<int>(data["t_max"]);
    int n_max  = as<int>(data["n_max"]);
    arma::mat u = Rcpp::as<arma::mat>(data["u"]);
    std::vector<int> gid  = data["gid"];
    // Create matrix to handle group data
    int n_group = 0;
    for( int i = 0; i < n_max; i++){if (n_group < gid[i]) n_group = gid[i];}
    std::vector<std::vector<int> >  g_mat(n_group, std::vector<int>(n_max));
    std::vector<int> g_count(n_group);
    for( int i = 0; i < n_max; i++){
        gid[i]--;
        g_mat[gid[i]][g_count[gid[i]]] = i;
        g_count[gid[i]]++;
    }
    Rcpp::Rcout << " Data input :" << " Checked" << std::endl;
    
    //Number of MCMC interations
    int iter  = as<int>(iter_);
    int numbatches  = as<int>(numbatches_);
    int batchlength  = as<int>(batchlength_);
    Rcpp::Rcout << " MCMC interations :" << " Checked" << std::endl;
    

    // Init hyperparams
    Rcpp::List init(init_);
    Rcpp::List other(other_);
    
    int core = other["core"];
    int disttype = other["disttype"];
    bool modelselect = other["modelselect"];
    arma::rowvec a  = init["a"];
    arma::rowvec b  = init["b"];
    arma::rowvec f0 = init["f0"];
    arma::vec  z = Rcpp::as<arma::vec>(init["z"]);
    arma::rowvec nu = zeros<rowvec>(n_group);      
    arma::mat zeta = zeros(t_max, n_group);
    arma::mat zeta_sqrt = zeros(t_max, n_group);
    arma::rowvec gamma = zeros<rowvec>(n_group);
    Rcpp::Rcout << " Core " << core << std::endl;
    Rcpp::Rcout << " Init hyperparams :" << " Checked" << std::endl;
    
    if (disttype > NORMDIST){
        nu = Rcpp::as<arma::rowvec>(init["nu"]);        
        zeta = Rcpp::as<arma::mat>(init["zeta"]);
        zeta_sqrt = sqrt(zeta);
    }
    if (disttype == HSSTDIST){
        gamma = Rcpp::as<arma::rowvec>(init["gamma"]);
    }

    Rcpp::Rcout << " disttype " << disttype << std::endl;
    
    // Init matrix
    // From init, generate f and rho, x
    
    arma::mat x_normal = zeros(t_max, n_max);
    arma::mat x_student = zeros(t_max, n_max);
    arma::mat x_normal_temp = zeros(t_max, n_max);
    arma::mat x_student_temp = zeros(t_max, n_max);
    arma::mat x_normal_DIC_post = zeros(t_max, n_max);
    arma::mat x_student_DIC_post = zeros(t_max, n_max);
    arma::mat x_normal_DIC_map = zeros(t_max, n_max);
    arma::mat x_student_DIC_map = zeros(t_max, n_max);
    
    arma::mat f = zeros(t_max, n_max);
    arma::mat f_temp = zeros(t_max, n_max);
    arma::rowvec f0_temp = f0;
    arma::rowvec f0_post(n_group);
    arma::rowvec f0_map(n_group); 
    
    arma::mat psi = zeros(t_max, n_max);
    arma::mat psi_temp = zeros(t_max, n_max);
    arma::mat rho = zeros(t_max, n_max);
    arma::mat rho_temp = zeros(t_max, n_max);
    
    arma::mat rho_DIC = zeros(t_max, n_max);
    arma::mat f_DIC = zeros(t_max, n_max);
    arma::mat psi_DIC = zeros(t_max, n_max);
    
    arma::mat f0_rec = zeros(n_group,iter);
    arma::mat a_rec = zeros(n_group,iter);
    arma::mat b_rec = zeros(n_group,iter);
    arma::mat nu_rec = zeros(n_group,iter);
    arma::mat gamma_rec = zeros(n_group,iter);
    arma::cube zeta_rec = zeros(t_max,n_group,iter);
    
    arma::mat LL_cond = zeros(t_max, n_max);
    arma::mat LL_cond_temp = zeros(t_max, n_max);
    arma::mat LL_cond_DIC = zeros(t_max, n_max);
    
    arma::rowvec sum_LL_cond(n_max);
    arma::rowvec sum_LL_cond_temp(n_max);
    arma::rowvec sum_LL_cond_DIC(n_max);
    
    arma::rowvec LL_rec(iter);
    arma::rowvec LL_rec_jwZ(iter);    
    
    arma::rowvec LL_rec_DIC4(iter, fill::zeros);
    arma::rowvec LL_rec_jwZ_DIC4(iter, fill::zeros);    
    arma::rowvec LL_rec_DIC6(iter, fill::zeros);
    arma::rowvec LL_rec_jwZ_DIC6(iter, fill::zeros);    
    
    arma::mat score_rec(t_max,iter);
    
    arma::mat psi_inv(n_max,n_max);
    arma::mat  Sigma(1,1);
    arma::rowvec me(n_max);
    arma::colvec mean_xt;
    arma::rowvec logsigma_f(n_group); logsigma_f.fill(-2); // add log_sigma move for f0
    arma::rowvec logsigma_a(n_group); logsigma_a.fill(-4); // add log_sigma move for a
    arma::rowvec logsigma_b(n_group); logsigma_b.fill(-4); // add log_sigma move for b
    arma::rowvec logsigma_nu(n_group); logsigma_nu.fill(0); // add log_sigma move for nu
    arma::rowvec logsigma_gamma(n_group); logsigma_gamma.fill(-1.5); // add log_sigma move for gamma
    
    
    
    arma::rowvec acount_f(n_group); acount_f.fill(0); // add count acceptance rate for f0
    arma::rowvec acount_a(n_group); acount_a.fill(0); // add count acceptance rate for a
    arma::rowvec acount_b(n_group); acount_b.fill(0); // add count acceptance rate for b
    arma::rowvec acount_nu(n_group); acount_nu.fill(0); // add count acceptance rate for nu
    arma::rowvec acount_gamma(n_group); acount_gamma.fill(0); // add count acceptance rate for gamma
    arma::mat acount_zeta(t_max,n_group); acount_zeta.fill(0); // add count acceptance rate for zeta
    
    
    
    
    double temp; double num_mh; double denum_mh; double alpha;
    double mode_target;  double cov_target;
    double nu_low = 2; // Lowest value of nu in student or skew student
    
    arma::rowvec a_temp = a; 
    arma::rowvec a_post(n_group); 
    arma::rowvec a_map(n_group); 
    
    arma::rowvec b_temp = b; 
    arma::rowvec b_post(n_group); 
    arma::rowvec b_map(n_group); 
    
    arma::mat zeta_temp = zeta;
    arma::mat zeta_sqrt_temp = zeta_sqrt;
    
    arma::rowvec nu_temp = nu;
    arma::rowvec nu_post(n_group); 
    arma::rowvec nu_map(n_group); 
    
    arma::rowvec gamma_temp = gamma;
    arma::rowvec gamma_post(n_group); 
    arma::rowvec gamma_map(n_group); 
    
    // With the init, generate the rho process
    if (disttype == NORMDIST) {
        for ( int i = 0; i < n_max; i++){
            for ( int t = 0; t < t_max; t++){
                x_normal(t,i) = R::qnorm(u(t,i),0,1,true,false);
            }
        }
        gen_rho(x_student,x_normal,u,z,rho,f,psi,gid,g_mat, g_count, a,b,f0,nu,gamma, zeta, zeta_sqrt,t_max,n_max,n_group, disttype,false);
        x_normal_DIC_post = x_normal;
        x_normal_DIC_map = x_normal;
    }else{
        gen_rho(x_student,x_normal,u,z,rho,f,psi,gid,g_mat, g_count, a,b,f0,nu,gamma, zeta, zeta_sqrt,t_max,n_max,n_group, disttype,true);        
        if (disttype == HSSTDIST) { nu_low = 6;}
        x_student_DIC_post = x_student;
        x_normal_DIC_post = x_normal;
        x_student_DIC_map = x_student;
        x_normal_DIC_map = x_normal;
    }

    double initLL = likelihood(x_normal,z,rho,t_max,n_max, LL_cond, sum_LL_cond, false);
    rho_temp.row(0) = rho.row(0);
    f_temp.row(0) = f.row(0);

    // Set parallel    
    #ifdef _OPENMP
        omp_set_num_threads(core);
    #endif
    
    
    Rcpp::Rcout << "Beginning MCMC...\n\n" << std::endl;
    int i=0;
    
    clock_t t;
    
    for( int numbat = 0; numbat < numbatches; numbat++){
        for( int batlen = 0; batlen < batchlength; batlen++){
            // Gen f0 f0_block
            #pragma omp parallel for default(none) private(num_mh,denum_mh,alpha,temp) shared(t_max,n_max,x_student,x_normal,u,z,rho,rho_temp,zeta,zeta_sqrt,logsigma_f,f0,f0_temp,f0_rec,f,f_temp,psi,psi_temp,i,a,b,nu,gamma,acount_f,gid,n_group,g_mat,g_count,LL_cond,LL_cond_temp,sum_LL_cond,sum_LL_cond_temp,disttype)
            for( int k = 0; k < n_group; k++){
                // Random walk for each of f_0
                f0_temp[k] = f0[k] + exp(logsigma_f[k])*R::rnorm(0,1);
                
                if ((fabs(f0_temp[k]) < 5) && (f0_temp[0] >0)){
                    // Recalculate rho_temp
                    gen_rho_paralG(x_student,x_normal,u,z,rho_temp,f_temp,psi_temp,gid,g_mat[k],a,b,f0_temp,nu,gamma, zeta, zeta_sqrt,t_max,n_max,k,g_count[k],disttype, false);
                    
                    // Compare the likelihood
                    num_mh = likelihood_group(x_normal,z, rho_temp, gid, g_mat[k], t_max,n_max, k,g_count[k], LL_cond_temp, sum_LL_cond_temp, true);
                    denum_mh = likelihood_group(x_normal,z, rho, gid, g_mat[k], t_max, n_max, k,g_count[k], LL_cond, sum_LL_cond, false);
                    alpha = num_mh - denum_mh;
                    //if (denum_mh == std::numeric_limits<double>::infinity()) throw("Error");
                    // Reject or Accept
                    temp = log(R::runif(0,1));
                    if (alpha > temp){
                        f0[k] = f0_temp[k];
                        f0_rec.col(i)[k] = f0[k];
                        rho.col(k) = rho_temp.col(k);
                        LL_cond.col(k) = LL_cond_temp.col(k);
                        sum_LL_cond[k] = sum_LL_cond_temp[k];
                        acount_f[k]++;
                    } else {
                        f0_rec.col(i)[k] = f0[k];
                        f0_temp[k] = f0[k];
                    }
                } else{
                    f0_rec.col(i)[k] = f0[k];
                    f0_temp[k] = f0[k];
                } 
            }
            
            rho_temp.row(0) = rho.row(0);
            f_temp.row(0) = f0toblock(f0, gid, t_max, n_max, n_group);
            f.row(0) = f0toblock(f0, gid, t_max, n_max, n_group);
            
            // Gen a
            #pragma omp parallel for default(none) private(num_mh,denum_mh,alpha,temp) shared(t_max,n_max,x_student,x_normal,u,z,rho,rho_temp,zeta,zeta_sqrt,nu,gamma,logsigma_a,f0,f,f_temp,psi,psi_temp,i,a,a_temp,a_rec,b,acount_a,gid,n_group,g_mat,g_count,LL_cond,LL_cond_temp,sum_LL_cond,sum_LL_cond_temp,disttype)
            for( int k = 0; k < n_group; k++){
                
                a_temp(k) = a(k) + exp(logsigma_a[k])*R::rnorm(0,1);
                if ((a_temp(k) > 0) && (a_temp(k) < 0.3) ){
                    gen_rho_paralG(x_student,x_normal,u,z,rho_temp,f_temp,psi_temp,gid,g_mat[k],a_temp,b,f0,nu,gamma, zeta, zeta_sqrt,t_max,n_max,k,g_count[k],disttype, false);
                    // Compare the likelihood
                    num_mh = likelihood_group(x_normal,z, rho_temp, gid, g_mat[k], t_max,n_max, k,g_count[k], LL_cond_temp,sum_LL_cond_temp, true);
                    denum_mh = likelihood_group(x_normal,z, rho, gid, g_mat[k], t_max, n_max, k,g_count[k], LL_cond, sum_LL_cond, false);
                    alpha = num_mh - denum_mh;
                    //if (denum_mh == std::numeric_limits<double>::infinity()) throw("Error");
                    temp = log(R::runif(0,1));
                    //if (k == 0 && i < 100) std::cout <<" The  num_mh " << num_mh << " a_temp" << a_temp << " denum_mh " << denum_mh << " alpha " << alpha << " b " << b <<  " temp " <<  temp << std::endl;
                    
                    if (alpha > temp){
                        a_rec(k,i) = a_temp(k);
                        a(k) = a_temp(k);
                        for( int g = 0; g < g_count[k]; g++){
                            rho.col(g_mat[k][g]) = rho_temp.col(g_mat[k][g]);
                            LL_cond.col(g_mat[k][g]) = LL_cond_temp.col(g_mat[k][g]);
                            sum_LL_cond[g_mat[k][g]] = sum_LL_cond_temp[g_mat[k][g]];
                        }
                        
                        acount_a[k]++;
                    } else {
                        a_rec(k,i) = a(k);
                    }
                    
                } else {
                    a_rec(k,i) = a(k);
                }
                    
                
            }
            
            // Gen b
            #pragma omp parallel for default(none) private(num_mh,denum_mh,alpha,temp) shared(t_max,n_max,x_student,x_normal,u,z,rho,rho_temp,zeta,zeta_sqrt,nu,gamma,logsigma_b,f0,f,f_temp,psi,psi_temp,i,b,b_temp,b_rec,a,acount_b,gid,n_group,g_mat,g_count,LL_cond,LL_cond_temp,sum_LL_cond,sum_LL_cond_temp,disttype)
            for( int k = 0; k < n_group; k++){
                b_temp(k) = b(k) + exp(logsigma_b[k])*R::rnorm(0,1);
                if ((b_temp(k) < 1 ) && (b_temp(k) > 0)){
                    //          Rcpp::Rcout << " This is group " << k << " " << " b_k" <<  b(k) << " b_ktemp" <<  b_temp(k) << std::endl;
                    //           Rcpp::Rcout << " The  rho[group_k]" << rho(0,g_mat[k][0]) << " " << rho(0,g_mat[k][1]) <<  " " << rho(0,g_mat[k][2]) << " " <<  rho(0,g_mat[k][3]) << std::endl;
                    //           Rcpp::Rcout << " The  rho[group_k]" << rho(t_max-1,g_mat[k][0]) << " " << rho(t_max-1,g_mat[k][1]) <<  " " << rho(t_max-1,g_mat[k][2]) << " " <<  rho(t_max-1,g_mat[k][3]) << std::endl;
                    
                    gen_rho_paralG(x_student,x_normal,u,z,rho_temp,f_temp,psi_temp,gid,g_mat[k],a,b_temp,f0,nu,gamma, zeta, zeta_sqrt,t_max,n_max,k,g_count[k],disttype, false);
                    
                    // Rcpp::Rcout << " The  rhoTE[group_k]" << rho_temp(t_max-1,g_mat[k][0]) << " " << rho_temp(t_max-1,g_mat[k][1]) <<  " " << rho_temp(t_max-1,g_mat[k][2]) << " " <<  rho_temp(t_max-1,g_mat[k][3]) << std::endl;
                    
                    // Compare the likelihood
                    num_mh = likelihood_group(x_normal,z, rho_temp, gid, g_mat[k], t_max,n_max, k,g_count[k], LL_cond_temp, sum_LL_cond_temp, true);
                    denum_mh = likelihood_group(x_normal,z, rho, gid, g_mat[k], t_max, n_max, k,g_count[k], LL_cond,sum_LL_cond, false);
                    alpha = num_mh - denum_mh;
                    //if (denum_mh == std::numeric_limits<double>::infinity()) throw("Error");
                    
                    temp = log(R::runif(0,1));
                    //if (k == 0 && i < 100) std::cout << " The  num_mh " << num_mh << " b_temp" << b_temp << " denum_mh " << denum_mh << " b " << b <<  " alpha " << alpha << " temp " <<  temp << std::endl;
                    
                    if (alpha > temp){
                        b_rec(k,i) = b_temp(k);
                        b(k) = b_temp(k);
                        for( int g = 0; g < g_count[k]; g++){
                            rho.col(g_mat[k][g]) = rho_temp.col(g_mat[k][g]);
                            LL_cond.col(g_mat[k][g]) = LL_cond_temp.col(g_mat[k][g]);
                            sum_LL_cond[g_mat[k][g]] = sum_LL_cond_temp[g_mat[k][g]];
                        }
                        acount_b[k]++;
                    } else {
                        b_rec(k,i) = b(k);
                    }
                    
                } else {
                    b_rec(k,i) = b(k);
                }
                
            }
            
            #pragma omp parallel for default(none) private(psi_inv,Sigma,me,mean_xt) shared(t_max,rho,n_max,x_normal,z)
            for( int j = 0; j < t_max; j++){
                psi_inv =  diagmat(1/(ones(1,n_max) - pow(rho.row(j),2)));
                
                Sigma = 1/(1 +rho.row(j) * psi_inv * rho.row(j).t() );
                me = Sigma * rho.row(j) * psi_inv;
                mean_xt = me * x_normal.row(j).t() ;
                z[j] = R::rnorm( (mean_xt[0]) , sqrt(Sigma[0]));
            }
            
            gen_rho(x_student,x_normal,u,z,rho,f,psi,gid,g_mat,g_count,a,b,f0,nu,gamma,zeta,zeta_sqrt,t_max,n_max,n_group,disttype,false);
            initLL = likelihood(x_normal,z,rho,t_max,n_max, LL_cond, sum_LL_cond, false);
            
            score_rec.col(i) =  z;

            if (disttype > NORMDIST){
                //Gen nu in groups
                #pragma omp parallel for default(none) private(num_mh,denum_mh,alpha,temp) shared(x_student,x_normal,x_student_temp,x_normal_temp,u,z,rho,rho_temp,f0,f,f_temp,psi,psi_temp,gid,g_mat,i,b,a,nu,logsigma_nu,nu_temp,acount_nu,nu_rec,gamma,logsigma_gamma,gamma_temp,acount_gamma,gamma_rec,zeta, zeta_sqrt, t_max,n_max,g_count,n_group,LL_cond,LL_cond_temp,sum_LL_cond,sum_LL_cond_temp,nu_low, disttype)
                for (int k = 0; k < n_group; k++){
                    
                    nu_temp[k] = nu[k] + exp(logsigma_nu[k])*R::rnorm(0,1);
                    //nu_temp[k] = nu[k] ;
                    
                    if (nu_temp[k] > nu_low && nu_temp[k] < 100){
                        gen_rho_paralG(x_student_temp, x_normal_temp,u,z,rho_temp,f_temp,psi_temp,gid,g_mat[k],a,b,f0,nu_temp,gamma,zeta, zeta_sqrt, t_max,n_max,k,g_count[k],disttype, true);
                        
                        num_mh = 0; denum_mh = 0;
                        num_mh = likelihood_group(x_normal_temp,z, rho_temp, gid, g_mat[k], t_max,n_max, k,g_count[k], LL_cond_temp, sum_LL_cond_temp, true);
                        num_mh += log_InvGammaG(zeta,nu_temp(k)/2, t_max, n_max, k) - log_studentG(x_student_temp, gid, g_mat[k],nu_temp[k],gamma[k],t_max, n_max, k,g_count[k], disttype);

                        denum_mh = likelihood_group(x_normal,z, rho, gid, g_mat[k], t_max, n_max, k,g_count[k], LL_cond,sum_LL_cond, false);
                        denum_mh += log_InvGammaG(zeta,nu(k)/2, t_max, n_max, k) - log_studentG(x_student, gid, g_mat[k],nu[k],gamma[k],t_max, n_max, k,g_count[k], disttype);

                        alpha = num_mh - denum_mh;
                        temp = log(R::runif(0,1));
                        //if (k < 1) Rcpp::Rcout << "Step 3, The  num_mh " << num_mh << " nu_temp" << nu_temp << " denum_mh " << denum_mh << " nu " << nu <<  " alpha " << alpha << " temp " <<  temp << std::endl;
                        
                        if (alpha > temp){
                            nu_rec(k,i) = nu_temp(k);
                            nu(k) = nu_temp(k);
                            
                            for( int g = 0; g < g_count[k]; g++){
                                x_student.col(g_mat[k][g]) = x_student_temp.col(g_mat[k][g]);
                                x_normal.col(g_mat[k][g]) = x_normal_temp.col(g_mat[k][g]);
                                // Remember to check if psi is ok or not
                                psi.col(g_mat[k][g]) = psi_temp.col(g_mat[k][g]);
                                rho.col(g_mat[k][g]) = rho_temp.col(g_mat[k][g]);
                                LL_cond.col(g_mat[k][g]) = LL_cond_temp.col(g_mat[k][g]);
                                sum_LL_cond[g_mat[k][g]] = sum_LL_cond_temp[g_mat[k][g]];
                            }
                            acount_nu[k]++;
                            
                        } else {
                            nu_rec(k,i) = nu(k);
                            
                        }
                        
                    } else {
                        nu_rec(k,i) = nu(k);
                    }
                    
                }
                
                
                if (disttype == HSSTDIST){
                    //Gen gamma in groups
                    #pragma omp parallel for default(none) private(num_mh,denum_mh,alpha,temp) shared(x_student,x_normal,x_student_temp,x_normal_temp,u,z,rho,rho_temp,f0,f,f_temp,psi,psi_temp,gid,g_mat,i,b,a,nu,logsigma_nu,nu_temp,acount_nu,nu_rec,gamma,logsigma_gamma,gamma_temp,acount_gamma,gamma_rec,zeta, zeta_sqrt, t_max,n_max,g_count,n_group,LL_cond,LL_cond_temp,sum_LL_cond,sum_LL_cond_temp,disttype)
                    for (int k = 0; k < n_group; k++){
                        
                        gamma_temp[k] = gamma[k] + exp(logsigma_gamma[k])*R::rnorm(0,1); // The correlation of MCMC goes around -0.3 to -0.2 so it does not worth to make a correlation
                        //gamma_temp[k] = gamma[k];
                        
                        gen_rho_paralG(x_student_temp, x_normal_temp,u,z,rho_temp,f_temp,psi_temp,gid,g_mat[k],a,b,f0,nu,gamma_temp,zeta, zeta_sqrt, t_max,n_max,k,g_count[k],disttype, true);
                        
                        num_mh = 0; denum_mh = 0;
                        num_mh = likelihood_group(x_normal_temp,z, rho_temp, gid, g_mat[k], t_max,n_max, k,g_count[k], LL_cond_temp, sum_LL_cond_temp, true);
                        num_mh += - log_studentG(x_student_temp, gid, g_mat[k],nu[k],gamma_temp[k],t_max, n_max, k,g_count[k], disttype);
                        
                        denum_mh = likelihood_group(x_normal,z, rho, gid, g_mat[k], t_max, n_max, k,g_count[k], LL_cond,sum_LL_cond, false);
                        denum_mh += - log_studentG(x_student, gid, g_mat[k],nu[k],gamma[k],t_max, n_max, k,g_count[k], disttype);
                        
                        alpha = num_mh - denum_mh;
                        temp = log(R::runif(0,1));
                        //if (i < 10) Rcpp::Rcout << "Step 3, The  num_mh " << num_mh << " gamma_temp" << gamma_temp << " denum_mh " << denum_mh << " gamma " << gamma <<  " alpha " << alpha << " temp " <<  temp << std::endl;
                        
                        if (alpha > temp){
                            gamma_rec(k,i) = gamma_temp(k);
                            gamma(k) = gamma_temp(k);
                            
                            for( int g = 0; g < g_count[k]; g++){
                                x_student.col(g_mat[k][g]) = x_student_temp.col(g_mat[k][g]);
                                x_normal.col(g_mat[k][g]) = x_normal_temp.col(g_mat[k][g]);
                                // Remember to check if psi is ok or not
                                psi.col(g_mat[k][g]) = psi_temp.col(g_mat[k][g]);
                                rho.col(g_mat[k][g]) = rho_temp.col(g_mat[k][g]);
                                LL_cond.col(g_mat[k][g]) = LL_cond_temp.col(g_mat[k][g]);
                                sum_LL_cond[g_mat[k][g]] = sum_LL_cond_temp[g_mat[k][g]];
                            }
                            acount_gamma[k]++;
                            
                        } else {
                            gamma_rec(k,i) = gamma(k);
                            
                        }
                        
                        
                    }
                }           
                
                // Gen zeta
                #pragma omp parallel for default(none) private(num_mh,denum_mh,alpha,temp,mode_target,cov_target) shared(t_max,n_max,x_student,x_normal,x_normal_temp,z,rho,rho_temp,zeta,zeta_sqrt,zeta_temp,zeta_sqrt_temp,nu,gamma,f0,f,f_temp,psi,psi_temp,i,zeta_rec,a,b,acount_zeta,gid,n_group,g_mat,g_count,LL_cond,LL_cond_temp,disttype)
                for (int k = 0; k < n_group; k++){
                    for( int j = 0; j < t_max; j++){

                        find_target(x_student, x_normal,z, rho, gid,g_mat[k], zeta, zeta_sqrt, nu(k), t_max, n_max, k, j, g_count[k], mode_target, cov_target);

                        double ran_stu = R::rt(4);
                        double log_zeta_temp = mode_target + sqrt(cov_target)*ran_stu;

                        zeta_temp(j,k) = exp(log_zeta_temp);
                        zeta_sqrt_temp(j,k) = sqrt(zeta_temp(j,k));

                        // Compare the likelihood

                        num_mh = likelihood_group_t(x_student, x_normal_temp,z,rho, gid,g_mat[k], gamma, zeta_temp,zeta_sqrt_temp,
                                                    t_max,n_max, k,j, g_count[k], LL_cond_temp, true);
                        //if (k == 0 && j == 105) std::cout << " The  num_mh1 " << num_mh << std::endl;
                        //num_mh += -log(zeta_temp(j,k))*(g_count[k]/2 + nu(k)/2) - nu(k)/2/zeta_temp(j,k) - R::dt(ran_stu, 4, true); //Becuase g_count[k] +1 is the number of dimension
                        num_mh += -0.5 * log(zeta_temp(j,k))*(g_count[k] + nu(k) ) - nu(k)/2/zeta_temp(j,k) - logStudentDensity(log_zeta_temp, mode_target, cov_target, 4); //Becuase g_count[k] +1 is the number of dimension
                        //if (k == 0 && j == 105) std::cout << " The  num_mh2 " << - log(zeta_temp(j,k))*(g_count[k]/2 + nu(k)/2) - nu(k)/2/zeta_temp(j,k) << " nu " << nu(k) << std::endl;
                        //if (k == 0 && j == 105) std::cout << " The  num_mh3 " << - logStudentDensity(log_zeta_temp, mode_target, cov_target, 4) << std::endl;
                        denum_mh = likelihood_group_t(x_student, x_normal,z,rho, gid,g_mat[k], gamma, zeta,zeta_sqrt,
                                                      t_max,n_max, k,j, g_count[k], LL_cond, true);
                        //if (k == 0 && j == 105) std::cout << " The denum_mh1 " << denum_mh << std::endl;
                        denum_mh += - 0.5 * log(zeta(j,k))*(g_count[k] + nu(k)) - nu(k)/2/zeta(j,k) - logStudentDensity(log(zeta(j,k)), mode_target, cov_target, 4);


                        alpha = num_mh - denum_mh;
                        temp = log(R::runif(0,1));

                        if (alpha > temp){

                            zeta(j,k) = zeta_temp(j,k);
                            zeta_sqrt(j,k) = zeta_sqrt_temp(j,k);

                            for( int g = 0; g < g_count[k]; g++){
                                x_normal(j,g_mat[k][g]) = x_normal_temp(j,g_mat[k][g]);
                                LL_cond(j,g_mat[k][g]) = LL_cond_temp(j,g_mat[k][g]);
                                // Remember to check rho is ok or not
                                if (j < t_max-1){
                                    dlogc(x_normal,z, rho,f,psi,j,n_max,g_mat[k][g]);
                                    f(j+1,g_mat[k][g]) = (1-b(k))*f0(k) + a(k) * psi(j,g_mat[k][g]) + b(k) * f(j,g_mat[k][g]);
                                    if (f(j+1,g_mat[k][g]) < -5.5 || f(j,g_mat[k][g]) == -5.5 ) {f(j+1,g_mat[k][g]) = -5.5;}             //adding checking step
                                    if (f(j+1,g_mat[k][g]) > 5.5 || f(j,g_mat[k][g]) == 5.5 ) {f(j+1,g_mat[k][g]) = 5.5;}             //adding checking step
                                    rho(j+1,g_mat[k][g]) = (1-exp(-f(j+1,g_mat[k][g]) ))/(1+exp(-f(j+1,g_mat[k][g]) ));
                                }

                            }
                            acount_zeta(j,k)++;
                        } else {

                            for( int g = 0; g < g_count[k]; g++){
                                // Remember to check rho is ok or not
                                if (j < t_max-1){
                                    dlogc(x_normal,z, rho,f,psi,j,n_max,g_mat[k][g]);
                                    f(j+1,g_mat[k][g]) = (1-b(k))*f0(k) + a(k) * psi(j,g_mat[k][g]) + b(k) * f(j,g_mat[k][g]);
                                    if (f(j+1,g_mat[k][g]) < -5.5 || f(j,g_mat[k][g]) == -5.5 ) {f(j+1,g_mat[k][g]) = -5.5;}             //adding checking step
                                    if (f(j+1,g_mat[k][g]) > 5.5 || f(j,g_mat[k][g]) == 5.5) {f(j+1,g_mat[k][g]) = 5.5;}             //adding checking step
                                    rho(j+1,g_mat[k][g]) = (1-exp(-f(j+1,g_mat[k][g]) ))/(1+exp(-f(j+1,g_mat[k][g]) ));
                                }

                            }

                        }


                    }
                }
                
                // Remember to recal sum_LL at the end.
                zeta_rec.slice(i) = zeta;
                // rho_rec.slice(i) = rho;
                // psi_rec.slice(i) = psi;
                // f_rec.slice(i) = f;
                double temp_LL;
                
                #pragma omp parallel for default(none) private(temp_LL) shared(x_normal,z, rho, t_max,n_max, LL_cond, sum_LL_cond)
                for (int g =0; g < n_max; g++){
                    temp_LL = likelihood_uniseries(x_normal,z, rho, t_max,n_max, g, LL_cond, sum_LL_cond, true);
                }
                
            }
            
            if (modelselect){
                
                ///////////////////////////////////
                if ( i == 3*iter/4-1){
                    int first_row = 0;
                    int first_col = iter/4;
                    int last_row = n_group-1;
                    int last_col = 3*iter/4-1;
                    a_post = trans(mean(a_rec.submat(first_row, first_col, last_row, last_col ), 1));
                    b_post = trans(mean(b_rec.submat(first_row, first_col, last_row, last_col ), 1));
                    nu_post = trans(mean(nu_rec.submat(first_row, first_col, last_row, last_col ), 1));
                    gamma_post = trans(mean(gamma_rec.submat(first_row, first_col, last_row, last_col ), 1));
                    f0_post = trans(mean(f0_rec.submat(first_row, first_col, last_row, last_col ), 1));
                    
                    Rcpp::Rcout << "i " << i << " a_post " << a_post << std::endl; 
                    Rcpp::Rcout << "i " << i << " b_post " << b_post << std::endl; 
                    Rcpp::Rcout << "i " << i << " nu_post " << nu_post << std::endl; 
                    Rcpp::Rcout << "i " << i << " gamma_post " << gamma_post << std::endl; 
                    Rcpp::Rcout << "i " << i << " f0_post " << f0_post << std::endl; 
                    sleep(1);
                    ///////////////////////////////////
                    
                    double LL_post = LL_rec[iter/2];
                    int map_index = iter/2;
                    for (int k = iter/2; k< 3*iter/4-1; k++){
                        if (LL_post < LL_rec[k]) {
                            map_index = k;
                            LL_post = LL_rec[k];
                        }
                    }
                    a_map = trans(a_rec.col(map_index));
                    b_map = trans(b_rec.col(map_index));
                    nu_map = trans(nu_rec.col(map_index));
                    gamma_map = trans(gamma_rec.col(map_index));
                    f0_map = trans(f0_rec.col(map_index));
                    //z_temp = score_rec.col(map_index);
                    
                    Rcpp::Rcout << "map_index " << map_index << " a_map " << a_map << std::endl; 
                    Rcpp::Rcout << "LL_rec_jwZ " << LL_rec_jwZ(map_index) << " b_map " << b_map << std::endl; 
                    Rcpp::Rcout << "map_index " << map_index << " nu_map " << nu_map << std::endl; 
                    Rcpp::Rcout << "map_index " << map_index << " gamma_map " << gamma_map << std::endl; 
                    Rcpp::Rcout << "map_index " << map_index << " f0_map " << f0_map << std::endl; 
                    sleep(1);
                    gen_rho(x_student_DIC_post,x_normal_DIC_post,u,z,rho_DIC,f_DIC,psi_DIC,gid,g_mat, g_count, a_post,b_post,f0_post,nu_post,gamma_post, zeta, zeta_sqrt,t_max,n_max,n_group, disttype,true);
                    gen_rho(x_student_DIC_map,x_normal_DIC_map,u,z,rho_DIC,f_DIC,psi_DIC,gid,g_mat, g_count, a_map,b_map,f0_map,nu_map,gamma_map, zeta, zeta_sqrt,t_max,n_max,n_group, disttype,true);
                    
                }
                
                
                model_select_LL(x_student, x_normal, z, gid, g_mat, g_count, nu, gamma, zeta, sum_LL_cond, LL_rec, LL_rec_jwZ,
                                t_max, n_max, n_group, i, disttype);                
                
                if ( i > 3*iter/4-1){
                    gen_rho(x_student_DIC_post,x_normal_DIC_post,u,z,rho_DIC,f_DIC,psi_DIC,gid,g_mat, g_count, a_post,b_post,f0_post,nu_post,gamma_post, zeta, zeta_sqrt,t_max,n_max,n_group, disttype,false);
                    initLL = likelihood(x_normal_DIC_post,z,rho_DIC,t_max,n_max, LL_cond_DIC, sum_LL_cond_DIC, false);
                    model_select_LL(x_student_DIC_post, x_normal_DIC_post, z, gid, g_mat, g_count, nu_post, gamma_post, zeta, sum_LL_cond_DIC, LL_rec_DIC4, LL_rec_jwZ_DIC4,
                                    t_max, n_max, n_group, i, disttype); 
                    
                    gen_rho(x_student_DIC_map,x_normal_DIC_map,u,z,rho_DIC,f_DIC,psi_DIC,gid,g_mat, g_count, a_map,b_map,f0_map,nu_map,gamma_map, zeta, zeta_sqrt,t_max,n_max,n_group, disttype,false);
                    initLL = likelihood(x_normal_DIC_map,z,rho_DIC,t_max,n_max, LL_cond_DIC, sum_LL_cond_DIC, false);
                    model_select_LL(x_student_DIC_map, x_normal_DIC_map, z, gid, g_mat, g_count, nu_map, gamma_map, zeta, sum_LL_cond_DIC, LL_rec_DIC6, LL_rec_jwZ_DIC6,
                                    t_max, n_max, n_group, i, disttype); 
                    
                }
                // Rcpp::Rcout << "i " << i << " LL_rec " << LL_rec(i) << " LL_rec_jwZ " << LL_rec_jwZ(i) << " LL_rec4 " << LL_rec_DIC4(i) << " LL_rec_jwZ4 " << LL_rec_jwZ_DIC4(i) << 
                //          " LL_rec6 " << LL_rec_DIC6(i) << " LL_rec_jwZ6 " << LL_rec_jwZ_DIC6(i) << " " << LL_rec_jwZ(i) - arma::accu(-0.5*(log(2 * M_PI) + pow(z,2) ) ) - LL_rec(i) << " " << arma::accu(-0.5*(log(2 * M_PI) + pow(z,2) ) ) << std::endl;
                
            }
            
            i++;
            
        }
        for (int jj=0; jj<n_group; jj++) {
            if (acount_f[jj] > batchlength * TARGACCEPT)
                logsigma_f[jj] = logsigma_f[jj] + adaptamount(numbat);
            else if (acount_f[jj] < batchlength * TARGACCEPT)
                logsigma_f[jj] = logsigma_f[jj] - adaptamount(numbat);
        }
        for (int jj=0; jj<n_group; jj++) {
            if (acount_a[jj] > batchlength * TARGACCEPT)
                logsigma_a[jj] = logsigma_a[jj] + adaptamount(numbat);
            else if (acount_a[jj] < batchlength * TARGACCEPT)
                logsigma_a[jj] = logsigma_a[jj] - adaptamount(numbat);
        }
        for (int jj=0; jj<n_group; jj++) {
            if (acount_b[jj] > batchlength * TARGACCEPT)
                logsigma_b[jj] = logsigma_b[jj] + adaptamount(numbat);
            else if (acount_b[jj] < batchlength * TARGACCEPT)
                logsigma_b[jj] = logsigma_b[jj] - adaptamount(numbat);
        }
        for (int jj=0; jj<n_group; jj++) {
            if (acount_nu[jj] > batchlength * TARGACCEPT)
                logsigma_nu[jj] = logsigma_nu[jj] + adaptamount(numbat);
            else if (acount_nu[jj] < batchlength * TARGACCEPT)
                logsigma_nu[jj] = logsigma_nu[jj] - adaptamount(numbat);
        }
        
        for (int jj=0; jj<n_group; jj++) {
            if (acount_gamma[jj] > batchlength * TARGACCEPT)
                logsigma_gamma[jj] = logsigma_gamma[jj] + adaptamount(numbat);
            else if (acount_gamma[jj] < batchlength * TARGACCEPT)
                logsigma_gamma[jj] = logsigma_gamma[jj] - adaptamount(numbat);
        }
        Rcpp::Rcout.precision(3);
        Rcpp::Rcout.setf(ios::fixed);
        Rcpp::Rcout << "i " << i << " a " ; a.raw_print(Rcout, "");
        Rcpp::Rcout << "i " << i << " b "; b.raw_print(Rcout, "");
        Rcpp::Rcout << "i " << i << " LL_rec " << LL_rec(i-1) << " LL_rec_jwZ " << LL_rec_jwZ(i-1) << " LL_rec4 " << LL_rec_DIC4(i-1) << " LL_rec_jwZ4 " << LL_rec_jwZ_DIC4(i-1) <<
            " LL_rec6 " << LL_rec_DIC6(i-1) << " LL_rec_jwZ6 " << LL_rec_jwZ_DIC6(i-1) << " " << LL_rec_jwZ(i-1) - arma::accu(-0.5*(log(2 * M_PI) + pow(z,2) ) ) - LL_rec(i-1) << " " << arma::accu(-0.5*(log(2 * M_PI) + pow(z,2) ) ) << std::endl;
        Rcpp::Rcout << "i " << i << " rho0 " << rho(0,0) << " r01 " << rho(0,1) << " r02 " << rho(0,2) << " r03 " << rho(0,3) << " r04 " << rho(0,4) << " r05 " << rho(0,5) << std::endl;
        Rcpp::Rcout.precision(2);
        Rcpp::Rcout.setf(ios::fixed);
        Rcpp::Rcout << "i " << i << " nu0 " ; nu.raw_print(Rcout, "");
        Rcpp::Rcout << "i " << i << " gamma0 "; gamma.raw_print(Rcout, "");
        //    Rcpp::Rcout << "i " << i << " LL " << LL_rec[i] << " zeta0 " << zeta(0,0) << " zeta01 " << zeta(0,1) << " zeta02 " << zeta(0,2) << " zeta03 " << zeta(0,3) << " zeta04 " << zeta(0,4) << " zeta05 " << zeta(0,5) << std::endl;
        Rcpp::Rcout << "i " << i << " logsigma_a " ; logsigma_a.raw_print(Rcout, "");
        Rcpp::Rcout << "i " << i << " logsigma_b " ; logsigma_b.raw_print(Rcout, "");
        Rcpp::Rcout << "i " << i << " logsigma_f " << logsigma_f[0] << " " << logsigma_f[1] << " " << logsigma_f[2]  << " " << logsigma_f[3] << " " << logsigma_f[4] << " " << logsigma_f[5]  << " " << logsigma_f[6] << " " << logsigma_f[7] << " " << logsigma_f[8] << " " << logsigma_f[9] <<  std::endl;
        Rcpp::Rcout << "i " << i << " logsigma_nu " ; logsigma_nu.raw_print(Rcout, "");
        Rcpp::Rcout << "i " << i << " logsigma_gamma " ; logsigma_gamma.raw_print(Rcout, "");
        Rcpp::Rcout << "i " << i << " x_student NA " << x_student.has_nan() << " max(gamma) " << max(gamma) << " min(gamma) " << min(gamma) << " max(f0) " << max(f0) << " min(f0) " << min(f0) << std::endl;
        //    Rcpp::Rcout << "i " << i << " acount_zeta " << acount_zeta(0,0) << " " << acount_zeta(0,1) << " " << acount_zeta(0,2)  << " " << acount_zeta(0,3) << " " << acount_zeta(0,4) << " " << acount_zeta(0,5)  << " " << acount_zeta(0,6) << " " << acount_zeta(0,7) << " " << acount_zeta(0,8) << " " << acount_zeta(0,9) <<  std::endl;
        
        acount_f.fill(0);
        acount_a.fill(0);
        acount_b.fill(0);
        acount_nu.fill(0);
        acount_gamma.fill(0);
        acount_zeta.fill(0);
    }
    t = clock() - t;
    std::cout << "It took " << ((double)t)/double(CLOCKS_PER_SEC) << " seconds.\n"  <<  std::endl;

    arma::rowvec DIC_mat(8,fill::zeros);
    arma::rowvec p_mat(8,fill::zeros);
    
    int num_params = n_group + 2 * n_group;    
    if (disttype == STUDDIST){ num_params = num_params + n_group;}
    if (disttype == HSSTDIST){ num_params = num_params + 2 * n_group;}
    double AIC = 0;
    double BIC = 0;
    
    if (modelselect){
        
        int first_row = 0;
        int first_col = 3*iter/4;
        int last_row = n_group-1;
        int last_col = iter-1;
        

        DIC_mat(3) = - 4 * mean( LL_rec_jwZ.subvec(first_col,last_col) ) + 2 * mean(LL_rec_jwZ_DIC4.subvec(first_col,last_col));
        p_mat(3) = -2 * (mean( LL_rec_jwZ.subvec(first_col,last_col) ) - mean(LL_rec_jwZ_DIC4.subvec(first_col,last_col)) );
        
        DIC_mat(7) = - 4 * mean( LL_rec.subvec(first_col,last_col) ) + 2 * mean(LL_rec_DIC4.subvec(first_col,last_col));
        p_mat(7) = -2 * (mean( LL_rec.subvec(first_col,last_col) ) - mean(LL_rec_DIC4.subvec(first_col,last_col)) );
        
        DIC_mat(4) = - 4 * mean( LL_rec_jwZ.subvec(first_col,last_col) ) + 2 * max(LL_rec_jwZ.subvec(first_col,last_col));
        p_mat(4) = -2 * (mean( LL_rec_jwZ.subvec(first_col,last_col) ) - max(LL_rec_jwZ.subvec(first_col,last_col)) );
        
        DIC_mat(5) = - 4 * mean( LL_rec_jwZ.subvec(first_col,last_col) ) + 2 * mean(LL_rec_jwZ_DIC6.subvec(first_col,last_col));
        p_mat(5) = -2 * (mean( LL_rec_jwZ.subvec(first_col,last_col) ) - mean(LL_rec_jwZ_DIC6.subvec(first_col,last_col)) );
        
        double LL_post = LL_rec_jwZ[first_col];
        int map_index = first_col;
        for (int k = first_col; k< last_col; k++){
            if (LL_post < LL_rec_jwZ[k]) {
                map_index = k;
                LL_post = LL_rec_jwZ[k];
            }
        }
        DIC_mat(6) = - 4 * mean( LL_rec.subvec(first_col,last_col) ) + 2 * LL_rec(map_index) ;
        p_mat(6) = -2 * (mean( LL_rec.subvec(first_col,last_col) ) - LL_rec(map_index)  );
        
        AIC = - 2 *  mean(LL_rec_DIC4.subvec(first_col,last_col)) + 2 * num_params;
        BIC = - 2 *  mean(LL_rec_DIC4.subvec(first_col,last_col)) + log(t_max) * num_params;     
        DIC_mat(0) = AIC;
        DIC_mat(1) = BIC;
        p_mat(0) = num_params;
    }
    
    List MCMCout            = List::create(Rcpp::Named("score_rec") = score_rec,
                                            Rcpp::Named("a_rec") = a_rec,
                                            Rcpp::Named("b_rec") = b_rec,
                                            Rcpp::Named("f0_rec") = f0_rec,
                                            Rcpp::Named("nu_rec") = nu_rec,
                                            //Rcpp::Named("zeta_rec") = zeta_rec,
                                            
                                            Rcpp::Named("rho_rec") = rho,
                                            Rcpp::Named("f_rec") = f,
                                            Rcpp::Named("psi_rec") = psi,
                                            Rcpp::Named("x_student") = x_student,
                                            Rcpp::Named("x_normal") = x_normal,
                                            Rcpp::Named("gamma_rec") = gamma_rec,
                                            
                                            Rcpp::Named("LL_rec") = LL_rec,
                                            Rcpp::Named("LL_rec_jwZ") = LL_rec_jwZ,
                                            Rcpp::Named("LL_rec_DIC4") = LL_rec_DIC4,
                                            Rcpp::Named("LL_rec_jwZ_DIC4") = LL_rec_jwZ_DIC4,
                                            Rcpp::Named("LL_rec_DIC6") = LL_rec_DIC6,
                                            Rcpp::Named("LL_rec_jwZ_DIC6") = LL_rec_jwZ_DIC6,
                                            Rcpp::Named("DIC") = DIC_mat,
                                            Rcpp::Named("p_DIC") = p_mat                                    
                                            //Rcpp::Named("num_params") = num_params,
                                            //Rcpp::Named("AIC") = AIC,
                                            //Rcpp::Named("BIC") = BIC 
                                        );
    return MCMCout;
    PutRNGstate();
    
    END_RCPP
}


// You can include R code blocks in C++ files processed with sourceCpp
// (useful for testing and development). The R code will be automatically
// run after the compilation.
//

/*** R
# main()
*/
