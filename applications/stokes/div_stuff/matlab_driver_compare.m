function matlab_driver_compare(input_path, output_path, xi, s_dim)
addpath('/Users/mattlowery/Desktop/code/matlab_master_rbffdcodes/Domains');
addpath('/Users/mattlowery/Desktop/code/matlab_master_rbffdcodes/Domains/MovingDomain');
evalc("[divs, max_abs_div, divs_i, max_abs_div_i, X, X_i, fs] = DriverDivCalcGeoFNO(input_path, 'div_stuff_verify', xi, s_dim);");
save(output_path, 'divs', 'max_abs_div', 'divs_i', 'max_abs_div_i', 'X', 'X_i', 'fs');
end
