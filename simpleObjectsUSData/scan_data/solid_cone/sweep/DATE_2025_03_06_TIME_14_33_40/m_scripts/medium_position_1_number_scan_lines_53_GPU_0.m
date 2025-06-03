% Simulating B-mode Ultrasound Scans for Simple Objects
% 
% This example illustrates how k-Wave can be used for the simulation of
% B-mode ultrasound images (including tissue harmonic imaging) analogous to
% those produced by a modern diagnostic ultrasound scanner. It builds on
% the Defining An Ultrasound Transducer, Simulating Ultrasound Beam
% Patterns, and Using An Ultrasound Transducer As A Sensor examples. 
%
% Note, this example generates a B-mode ultrasound image using 
% kspaceFirstOrder3DG. Compared to ray-tracing or Field II, this approach 
% is very general. In particular, it accounts for nonlinearity, multiple 
% scattering, power law acoustic absorption, and a finite beam width in the 
% elevation direction. However, it is also computationally intensive. Use a 
% modern GPU and the Parallel Computing Toolbox (with 'DataCast' set to 
% 'gpuArray-single') for significantly lower runtime instead of using a modern 
% desktop CPU (with 'DataCast' set to 'single'). 
% 
% Authored by Pratik Dash

%#ok<*UNRCH>

clearvars;

% =========================================================================
% GET INPUTS FROM THE PYTHON SCRIPT
% =========================================================================

% set the isotropic PML size
pml_size = 20; % [grid points]

% set the location to save mat files
data_path = 'C:\Users\PratikDash\Documents\UltrasoundSimulation\B-mode_simple_objects\tmp\solid_cone\sweep\DATE_2025_03_06_TIME_14_33_40\';

% set the location to save intermediate I/O files
io_path = 'C:\Users\PratikDash\Documents\UltrasoundSimulation\B-mode_simple_objects\tmp\solid_cone\sweep\DATE_2025_03_06_TIME_14_33_40\io_files\';

% set the name of the intermediate input and output data files
data_name = 'medium_position_1_number_scan_lines_53_GPU_0';

% set the location of the .mat file which will contain properties
% of the medium
mat_file = 'C:\Users\PratikDash\Documents\UltrasoundSimulation\B-mode_simple_objects\images\solid_cone.mat';

% set the isotropic grid spacing
ds = 6.35e-05; % [m]

% set the CFL number
cfl = 0.1;

% set the transducer properties
transducer_total_elements = 80;
transducer_active_elements = 1;
transducer_element_width = 4; % in [grid points]
transducer_element_length = 189; % in [grid points]
transducer_element_spacing = 0; % in [grid points]
transducer_radius = inf; % in [m]
transducer_focus_distance = 0.065; % in [m]
transducer_elevation_focus_distance = 0.065; % in [m]
transducer_steering_angle = 0; % in [degrees]
transducer_transmit_apodization = 'Rectangular'; % 'Rectangular' or 'Hanning'
transducer_receive_apodization = 'Rectangular'; % 'Rectangular' or 'Hanning'

% set inputs for the input signal
central_frequency = 2700000.0; % [Hz]
fwhm_bandwidth = 85 / 100 * central_frequency; % [Hz]
t0 = 1e-06; % in [s]

% set the simulation grid size without the PML
Nx = 216; % [grid points]
Ny = 4; % [grid points]
Nz = 216; % [grid points]

% set the active transducer position in the grid
active_transducer_x_axis_position = 1; % in [grid points]
active_transducer_y_axis_position = 1; % in [grid points]
active_transducer_z_axis_position = 15; % in [grid points]

% get the flag that decides if a sweep or a single acquisition is to be performed
sweep_flag = 0; % set to 0 for a sweep and 1 for a single acquisition

% set the starting position for the medium
medium_y_starting_position = 1; % in [grid points]

% set the number of scan lines
number_scan_lines = 53;

% simulation settings
data_cast = 'gpuArray-single'; % set to 'single' or 'gpuArray-single' to speed up computations

% =========================================================================
% DEFINE THE K-WAVE GRID
% =========================================================================

% set the grid spacing size between the grid points
dx = ds;
dy = ds;
dz = ds;

% create the k-space grid
kgrid = kWaveGrid(Nx, dx, Ny, dy, Nz, dz);

% =========================================================================
% DEFINE THE MEDIUM PROPERTIES
% =========================================================================

% define the properties of the propagation medium
c0 = 1540;                      % [m/s]
rho0 = 1000;                    % [kg/m^3]
% medium.alpha_coeff = 0.75;      % [dB/(MHz^y cm)]
medium.alpha_power = 1.01;
medium.BonA = 6;

% load the density and sound speed maps
object_data = load(mat_file); 
% make sure that all fields in the struct are converted to 'double'
object_data = structfun(@double, object_data, 'UniformOutput', false);
% create the 3D matrix of the object
object = zeros(1, prod(object_data.grid_size));
% assign the values to the indices
object(object_data.indices) = object_data.values;
% reshape the object to the grid size
object = reshape(object, object_data.grid_size);

% prepare density and sound speed maps
% prepare a list of unique values in object
unique_values = unique(object);
% initialize maps of the material properties
density_map = zeros(size(object));
sound_speed_map = zeros(size(object));
alpha_coeff_map = zeros(size(object));
% loop over the unique values
for i = 1:length(unique_values)
    % get the indices of the current value
    indices = object == unique_values(i);
    % get the number of voxels with the current value
    num_voxels = sum(indices(:));
    % Assign density and sound speed values
    % with 0.1% additional random Gaussian noise. 
    % Get separate random numbers for each unique value
    % and use them for both density and sound speed. 
    rand_numbers = randn(1, num_voxels);
    density_map(indices) = object_data.medium_properties(1, i) * (1 + 0.001 * rand_numbers);
    sound_speed_map(indices) = object_data.medium_properties(2, i) * (1 + 0.001 * rand_numbers);
    alpha_coeff_map(indices) = object_data.medium_properties(3, i) * (1 + 0.001 * rand_numbers);
end

% Check to see if the grid spacing of the imported object 
% is the same as the grid spacing of the simulation.
% If not the same, interpolate the object to the simulation grid
if object_data.grid_spacing(1) ~= dx*1e3 || object_data.grid_spacing(2) ~= dy*1e3 || object_data.grid_spacing(3) ~= dz*1e3
    % create the grid for the object data
    [xq, yq, zq] = meshgrid(linspace(0, (object_data.grid_size(2)-1)*object_data.grid_spacing(2), round(object_data.grid_size(2)*object_data.grid_spacing(2)*1e-3/dx)), ...
                            linspace(0, (object_data.grid_size(1)-1)*object_data.grid_spacing(1), round(object_data.grid_size(1)*object_data.grid_spacing(1)*1e-3/dy)), ...
                            linspace(0, (object_data.grid_size(3)-1)*object_data.grid_spacing(3), round(object_data.grid_size(3)*object_data.grid_spacing(3)*1e-3/dz)));

    % create the grid for the simulation
    [x, y, z] = meshgrid(0:object_data.grid_spacing(2):(object_data.grid_size(2)-1)*object_data.grid_spacing(2), ...
                         0:object_data.grid_spacing(1):(object_data.grid_size(1)-1)*object_data.grid_spacing(1), ...
                         0:object_data.grid_spacing(3):(object_data.grid_size(3)-1)*object_data.grid_spacing(3));
    % interpolate the density map
    density_map = interp3(x, y, z, density_map, xq, yq, zq, 'linear', 0);
    % interpolate the sound speed map
    sound_speed_map = interp3(x, y, z, sound_speed_map, xq, yq, zq, 'linear', 0);
    % interpolate the alpha coefficient map
    alpha_coeff_map = interp3(x, y, z, alpha_coeff_map, xq, yq, zq, 'linear', 0);
end

% save the size of the final grid 
N_tot = size(density_map);

% delete unnecessary variables
clear unique_values indices num_voxels i xq yq zq x y z;

% =========================================================================
% DEFINE THE INPUT SIGNAL
% =========================================================================

% first create the time array
t_end = (Nx * dx) * 2.2 / c0; % [s]
kgrid.makeTime(c0, cfl, t_end);

% define strength of the input signal
source_strength = 1e6;          % [Pa]

% calculate the standard deviation of the Gaussian envelope in time domain
sigma_t = 1 / (2 * pi * (fwhm_bandwidth / sqrt(8 * log(2))));  
% create the input signal
% Sine wave with a Gaussian envelope
input_signal = sin(2 * pi * central_frequency * (kgrid.t_array - t0)) .* ...
               exp(-0.5 * ((kgrid.t_array - t0) / sigma_t).^2);  
% correct the dimensions of the input signal before assigning it to 
% kWaveTransducer
input_signal = input_signal.'; % make the input signal a column vector

% scale the source magnitude by the source_strength divided by the
% impedance (the source is assigned to the particle velocity)
input_signal = (source_strength ./ (c0 * rho0)) .* input_signal;

% =========================================================================
% DEFINE THE ACTIVE ULTRASOUND TRANSDUCER
% =========================================================================

% physical properties of the active transducer
transducer.number_elements = transducer_active_elements; % total number of transducer elements
transducer.element_width = transducer_element_width; % width of each element [grid points]
transducer.element_length = transducer_element_length; % length of each element [grid points]
transducer.element_spacing = transducer_element_spacing; % spacing (kerf  width) between the elements [grid points]
transducer.radius = transducer_radius; % radius of curvature of the transducer [m]

% properties used to derive the beamforming delays
transducer.sound_speed = c0;                    % sound speed [m/s]
transducer.focus_distance = transducer_focus_distance; % focus distance [m]
transducer.elevation_focus_distance = transducer_elevation_focus_distance; % focus distance in the elevation plane [m]
transducer.steering_angle = transducer_steering_angle; % steering angle [degrees]

% apodization
transducer.transmit_apodization = transducer_transmit_apodization; % 'Rectangular' or 'Hanning'
transducer.receive_apodization = transducer_receive_apodization; % 'Rectangular' or 'Hanning'

% position the active transducer in the computational grid
transducer.position = round([active_transducer_x_axis_position, ...
                             active_transducer_y_axis_position, ...
                             active_transducer_z_axis_position]);

% define the transducer elements that are currently active
transducer.active_elements = ones(transducer.number_elements, 1);
% get the active transducer width in grid points
active_transducer_width = transducer_active_elements * transducer_element_width + ...
                          (transducer_active_elements - 1) * transducer_element_spacing;                   

% append input signal used to drive the transducer
transducer.input_signal = input_signal;

% create the transducer using the defined settings
transducer = kWaveTransducer(kgrid, transducer);

% print out transducer properties
transducer.properties;

% =========================================================================
% SAVE RELEVANT DATA
% =========================================================================

% give the name of the file to be saved
sim_spec_file = [data_path '\simulation_specifics.mat'];
% save it if it does not exist already
if ~isfile(sim_spec_file)
    % create a new struct and save relevant data relevant to the simulation
    data = {};
    data.time_array = kgrid.t_array;
    data.input_signal = input_signal;
    data.c0 = c0;
    data.rho0 = rho0;
    data.N = [Nx, Ny, Nz];
    data.N_tot = N_tot;
    data.d = [dx, dy, dz];
    % data.medium_alpha_coeff = medium.alpha_coeff;
    data.medium_alpha_power = medium.alpha_power;
    % save the data to data_path in a -v7.3 compatible format
    save([data_path '\simulation_specifics.mat'], '-struct', 'data', '-v7.3');
end

% =========================================================================
% RUN THE SIMULATION
% =========================================================================

% preallocate the storage
scan_lines = zeros(number_scan_lines, kgrid.Nt);

% set the input settings
input_args = {...
    'PMLInside', false, 'PMLSize', pml_size, 'DataPath', io_path, ...
    'DataName', data_name, ...
    'DataCast', data_cast, 'DataRecast', true, 'PlotSim', true};

% run the simulation by looping through the scan lines
for scan_line_index = 1:number_scan_lines    

    % update the command line status
    disp('');
    disp(['Computing scan line ' num2str(scan_line_index) ' of ' num2str(number_scan_lines)]);

    % get the current starting index of the medium
    medium_position = medium_y_starting_position + (scan_line_index - 1) * active_transducer_width;

    % load the current section of the medium
    medium.sound_speed = sound_speed_map(:, medium_position:medium_position + Ny - 1, :);
    medium.density = density_map(:, medium_position:medium_position + Ny - 1, :);
    medium.alpha_coeff = alpha_coeff_map(:, medium_position:medium_position + Ny - 1, :);
    
    % run the simulation
    sensor_data = kspaceFirstOrder3DG(kgrid, medium, transducer, transducer, input_args{:});
    % extract the scan line from the sensor data
    scan_lines(scan_line_index, :) = transducer.scan_line(sensor_data);

    % save this current scan line at data_path indexed by medium_position
    % Note that it must be in the format (something x kgrid.Nt)
    data = {};
    data.c0 = c0;
    data.scan_line = reshape(squeeze(scan_lines(scan_line_index, :)), [1, kgrid.Nt]);
    data.time_array = reshape(kgrid.t_array, [1, kgrid.Nt]);
    data.sensor_data = reshape(squeeze(sensor_data), [transducer.number_elements, kgrid.Nt]);
    % save as a -v7.3 struct
    save([data_path '\scan_line_medium_position=' num2str(medium_position) '.mat'], '-struct', 'data', '-v7.3');
end

