data {					    // Data block
  int<lower=0> N;			// Input is an integer N
  real r[N];				// Input is a real vector r[N], it could be written as vector r[N] also
}
parameters {				// Parameter block where stan records their value and makes inference on.
  real mu;					// mu: parameter c of the process
  real<lower=0,upper=5> alpha0;			    // alpha0 is omega in the model 
  real<lower=0,upper=1> alpha1;	    // alpha1 is restricted in (0,1)
  real<lower=0,upper=(1-alpha1)> beta1;     // beta1 is restricted in (0,alpha1)
  real<lower=4,upper=30> nu;		// nu is the degree of freedom
  real<lower=0,upper = 5> sigma1;			    // Initial value for sigma
  real<lower=0> zeta[N];            // Mixing value
  real gamma;                       // skewness parameter
  real<lower=-1,upper=1> phi1;      // AR(1) for conditional mean
}	
transformed parameters {		// block of transformed parameters where in each leapfrog will be executed
  real<lower=0> sigma[N];		// we record a vector of sigma
  real mean_skew;
  real sd_skew;
  
  mean_skew = gamma * nu/(nu - 2);      //We calculate the mean and variance of the HSST dist
  sd_skew = sqrt( 2 * pow(gamma,2) * pow(nu,2)/pow(nu-2,2)/(nu-4) + nu/(nu-2)  ); 
  
  
  sigma[1] = sigma1;			// at t = 1, sigma[1] = sigma1 the parameter
  sigma[2] = sqrt(alpha0
                     + alpha1 * pow(r[1] - mu/(1 - phi1), 2)			// pow() is the power function, it could be ^ instead
                     + beta1 * pow(sigma[1], 2));
  
  for (t in 3:N)				// Calculate sigma[t]^2 as a ARMA process
    {   
        sigma[t] = sqrt(alpha0
                     + alpha1 * pow(r[t-1] - mu - phi1 * r[t-2], 2)			// pow() is the power function, it could be ^ instead
                     + beta1 * pow(sigma[t-1], 2));
    }
      
    
}
model {					// model block, if not specify, parameters are uniform distributed.
  nu ~ gamma(2,0.5);			// prior for nu
  sigma1 ~ normal(0,1);
  gamma ~ normal(0,1);
  zeta ~ inv_gamma(nu/2,nu/2);	
  for (t in 2:N)				// Calculate sigma[t]^2 as a ARMA process
    {
        target += normal_lpdf(r[t] | mu + phi1 * r[t-1] + (-mean_skew + gamma * zeta[t])*sigma[t]/sd_skew   , sqrt( zeta[t] )*sigma[t]/sd_skew );
    }

}
