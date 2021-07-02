function out = astro_prior_sample()


  om_prior      = betarnd(3,3);
  w0_prior      = normrnd(-0.5,0.5);

out = [om_prior,w0_prior];
end
