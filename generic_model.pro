pro generic_model, Time_range_this_run = Time_range_this_run, test_particle_this_run = test_particle_this_run, line_this_run = line_this_run, $
  Observatory_this_run = Observatory_this_run, Output_title_this_run = Output_title_this_run, UTC_this_run = UTC_this_run, $
  Speed_Distribution_this_run = Speed_Distribution_this_run, Surface_Distribution_this_run = Surface_Distribution_this_run, Loop_times_this_run = Loop_times_this_run, $
  Upward_flux_at_exobase_this_run = Upward_flux_at_exobase_this_run

  ; Carl Schmidt, BU, 2020
  ; Originally adapted from Jody Wilson, Boston University, 2007
  ;
  ; MODIFICATIONS:
  ;    5/7/2020: began edits to output display for MESSENGER UVVS viewing comparisons
  ;
  ; DEPENDENCIES:
  ;
  ; 1. This code implents routines from the Coyote, and NASA IDL Astro Libraries. These may be downloaded at:
  ;    http://www.idlcoyote.com/documents/programs.php#COYOTE_LIBRARY_DOWNLOAD
  ;    http://idlastro.gsfc.nasa.gov/ftp/
  ;    Both must be present in the IDL path directories
  ;
  ; 2. The ICY version of SPICE must be compiled and linked. The procedure under Windows OS is included as
  ;    a word doc in the main directory.
  ;
  ;****************************************PROCEDURES CALLED***************************************************************
  ;Procedures and functions used (tab = nested):

  ;LOAD_SPICE.pro - Load spice kernels

  ;LOAD_PARTICLE_DATA.pro - Load constants and cross-sections specific to the species modeled

  ;LOAD_LINE_DATA.pro - load emission line constants and a portion of a high resolution solar spectrum surrounding
  ;                     the emission line of interest.

  ;RELEASE_STATE.pro - sets the spatial particle release points, the initial velocity vectors are set to unity and
  ;                    then scaled by generate_velocity_distribution.pro

  ;GENERATE_VELOCITY_DISTRIBUTION.pro - gives launch velocities to the particles under a specified energy distribution.

  ;IRRADIANCE.pro - Extrapolates solar spectral flux from SORCE and SEE to the planetary body at the time UTC, if no
  ;                 Solar Irradiance Data can be retreived from the web API, default is Jan 1 2010.

  ;RK4_INTEGRATE.pro - 4th order Runge-Kutta integrator including, the body and Sun's gravity, and parent body gravity
  ;                    for satellites. Ionization with shadowing, and (only for Na) radiation acceleration with shadowing

  ;RADACCEL.pro - finds radiation acceleration (only for Na)

  ;GVALUE.pro - finds resonant scattering rate (Currently only for Na)

  ;ILLUMINATION.pro: - Calculates the solar illumination on a bunch of points in space, i.e. the umbra / penumbra

  ;OUTPUT_DISPLAY.pro - generates an image in Rayleighs with user defined platescaling and viewing perspective

  ;LOADGOOD.pro - makes a color table

  ;========================================COMMON BLOCKS===================================================================

  COMMON Model_shared, Body, Ephemeris_time, Seed, Directory, Particle_data, Line_data, Debug
  COMMON Output_shared, Plot_range, Output_Size_In_Pixels, Output_Title, Center_in_frame, viewpoint, FOV, N_ticks, Tickstep, Observatory, Above_Ecliptic, Boresight_Pixel, Aperture_Corners

  ;===========================================INPUTS=======================================================================
  if !VERSION.OS_FAMILY eq 'unix' then begin
    directory          = '/Users/tica9197/Documents/GitHub/Exosphere-Model/read_write/' ; Directory where all other files are read from and written to
    kernel_directory   = '/Users/tica9197/Documents/mercury/kernels/'                           ; Directory where the spice kernel files live
  endif
  if !VERSION.OS_FAMILY eq 'Windows' then begin
    directory          = 'C:\IDL\Generic Model V2\read_write\' ; Directory where all other files are read from and written to
    kernel_directory   = 'C:\SPICE\'                           ; Directory where the spice kernel files live
  endif
    Body               = 'Mercury'                             ; e.g., 'Mercury', 'CHURYUMOV-GERASIMENKO', 'Moon'
    ;UTC                = 'Feb 26, 2017 07:00:00'              ; Universal coordinate time to be modeled, time when image taken.
    ;UTC                = '2020-Nov-13 23:20'                   ; Mercury Peak Na radiation pressure
    ;UTC                = '2007-Jun-06 12:40'                   ; Mercury Aphelion
    ;UTC                = '2018-12-13T16:22:49'                ; Potassium conventional data from Haleakala
    ;UTC                = 'Apr 08, 2005 22:10:00'              ; Solar eclipse from McDonald
    if keyword_set(UTC_this_run) then UTC=UTC_this_run else $  ; Use the input specified by the caller, or...
      UTC                 = '2011-Aug-04 02:08:37.43'           ; Tim's first datapoint UTC

    if keyword_set(test_particle_this_run) then test_particle = test_particle_this_run else $
      test_particle      = 'Na'                                  ; Species to model
    ; Options: 'Na'
    ;          'Mg'
    ;          'K'
    if keyword_set(Line_this_run) then Line = Line_this_run else $
      Line               = 'Na-D'                                ; Emission line to model
    ; Options: 'Na-D'
    ;          'Mg-2853'
    ;          'K-D'
    if keyword_set(Speed_distribution_this_run) then Speed_distribution = Speed_distribution_this_run else $
      Speed_distribution = 'MBF_1200K'                           ; Set the speed distribution the particles will be released with
    ; Options: 'Maxwellian_3000K'  where temperature is variable
    ;          'MBF_3000K'         where temperature is variable,
    ;                              Maxwell-Bolzmann Flux Dist. (V^3)
    ;          'Kappa_1500K_K=1.8' where temperature is variable
    ;                              and ~1.6 < K < ~30
    ;          'Step_[Vmin,Vmax]'  random velocities between Vmin
    ;                                        and Vmax specified in km/s
    if keyword_set(Surface_distribution_this_run) then Surface_distribution = Surface_distribution_this_run else $
      Surface_distribution = 'Global' ;'Dayside'                           ; Set the surface distribution the particles will be released with
    ; Options: 'Global'            Everywhere uniform
    ;          'Dayside'           Uniform 2pi steradians, centered in sub-solar longitude [-90,90 lat]
    ;          'Point_[lon,lat]'   Specified W. Lon & Lat [degrees]
    viewpoint =          'MESSENGER'                           ; How view the output:
    ; Options: 'Insert Body Name' (SPICE recognized name, e.g. 'Earth' = Geocentric, or 'STEREO A')
    ;          'Moon Spot'        (from Observatory looking anti-sunward)
    ;Observatory =        'McDonald'                           ; Set observer's location as a named observatory
    ; Options: Anything location within ASTROLIB's Observatory.pro, e.g.,
    ;          'CASLEO'            El Leoncito, Argentina
    ;          'McDonald'          McDonald Observatory, TX, USA
    ;Observatory = Observatory_this_run
    Image_type             = 'Rayleighs_1.e3'                  ; Scaling for the postscript output. 'Rayleighs_1.eX' where, e.g., an X of 3 corresponds to kilorayleighs
    if keyword_set(Output_title_this_run) then Output_title=Output_title_this_run else $
      Output_title           = 'Test'                            ; Title of the model's output FITS and Postcript files
    ; FITS extension 1 contains the metadata with the input parameters.
    Number_of_particles    = long(2^15-1)                      ; Number of packets of atoms in the simulation, do not exceed 32766 = 2^15-1
    if keyword_set(Time_range_this_run) then Time_range=Time_range_this_run else $
      Time_range             = [0.,.166]                         ; When to start and stop releasing particles from the planet [End, Begin] days ago
    ;                                                           ; A single element array represents a 1 sec 'pulse' of particles occuring [pulse_time] days ago
    timestep               = 50.                               ; Time step in seconds from the RK4 integrator, use < = 50 s at Mercury
    Plot_range             = 1600.                             ; for 'Above ecliptic' viewing only. Defines the plate scale, the spatial distance of each axis to be plotted in BODY RADII
    Output_Size_In_Pixels  = [128., 128., 128.]                ; Number of pixels on the [x,y,z] axis of the plot window (keep them all the same for sanity)
    Center_in_frame        = [1./2., 1./2., 1./2.]             ; [x,y,z] position of Body center within the output field of view, [1./2., 1./2., 1./2.] = centered
    ;Upward_flux_at_exobase = 1.e26                             ; Upward Flux accross the exobase in particles per second, integrated over 2 pi steradians
    if keyword_set(Upward_flux_at_exobase_this_run) then Upward_flux_at_exobase=Upward_flux_at_exobase_this_run else $
      Upward_flux_at_exobase = 1.e26                             ; Meteor's total vapourization
    ;Seed                   = 2653553L                         ; A large scalar long integer to seed the random number generator, comment out if the model is always to be initialized randomly.
    Debug                  = 0                                 ; Set to 1 to output more detail at intermidiate steps.
    exobase_height         = 0.d                               ; Exobase altitude above the surface, units of KM
    if keyword_set(Loop_times_this_run) then Loop_times=Loop_times_this_run else $
      Loop_times             = 1.                                ; How many loops the model should run, runs are stacked and averaged so this sets statistical noise
    FOV                    = 3600.*90.                         ; ARCSECONDS to a side. Field of View to display for output images that are in SKY COORDINATES
    N_ticks                = 10.                               ; Number of tick marks in the axis of the sky coordinate image
    tickstep               = 200.                              ; Axis tick step size in Body radii for the 'Above ecliptic' viewings

    ;===========================================KEYWORDS=======================================================================
    Bounce                 = 0                                 ; Particles re-impacting the surface can bounce (=1) or stick (=0)
    Label_Phase            = 0                                 ; Display the body's phase angle as pixels? MAJOR ISSUE: Needs rotation to plane of sky.
    Above_Ecliptic         = 0                                 ; Also writes an output image viewed from above the ecliptic plane with "Plot_range" field of view (longer run times)

    ;***********************************************OUTPUTS*************************************************
    ;                         All outputs are saved to the read_write directory
    ;
    ; Model_Image_CD (loop_number).sav    = the file imgxy, the column density a single integration run.
    ; Model_Image_R (loop_number).sav     = the file imgxy, the brightness a single integration run.
    ; (output)_Column_Density.ps          = Postscript plot of the column density averaged over all integration runs
    ; (output)_Emission.ps                = Postscript plot of the brightness averaged over all integration runs
    ;*******************************************************************************************************

    ; How long did the model take to execute? Record the time at which the program starts
    start_time = systime(/seconds)

    ;  ; Create an IDL structure containing both the final model result the inputs used.
    ;  ; Note: if the structure below is appended or edited, IDL must (!) be reset (IDL> .reset) to avoid conflicting tag definitions
    ;    Inputs = ['UTC','Time_range','Upward_flux_at_exobase','Speed_distribution','FOV','Output_Size_In_Pixels']
    ;    Model_Results_Structure = CREATE_STRUCT(Name = 'Inputs_Used', Inputs, UTC, Time_range, Upward_flux_at_exobase, $
    ;                                            Speed_distribution, FOV, Output_Size_In_Pixels)

    ; Define some data cubes. Each loop will write an image to a layer of these. The they're averaged into final output images
    initialize_array = fltarr(Output_Size_In_Pixels[0], Output_Size_In_Pixels[1], loop_times)  ; Reform needed as a work-around in case of loop_times = 1
    Model_Cube_R     = reform(initialize_array, Output_Size_In_Pixels[0], Output_Size_In_Pixels[1], loop_times) ; stack/cube of the output images in Rayleighs
    Model_Cube_CD    = reform(initialize_array, Output_Size_In_Pixels[0], Output_Size_In_Pixels[1], loop_times) ; stack/cube of the output images in cgs column density
    if keyword_set(Above_Ecliptic) then Above_Ecliptic_Cube_R  = reform(initialize_array, Output_Size_In_Pixels[0], Output_Size_In_Pixels[1], loop_times)
    if keyword_set(Above_Ecliptic) then Above_Ecliptic_Cube_CD = reform(initialize_array, Output_Size_In_Pixels[0], Output_Size_In_Pixels[1], loop_times)

    ; Load all ephemeride data
    LOAD_SPICE, Kernel_Directory

    ; Get the cartesian state vector of the body at UTC in units of body radii / body radii per second
    cspice_bodn2c, Body, planet_id, found       ; Look up the integer code for the target body. Found must equal 1 for valid SPICE objects
    cspice_str2et, UTC, ephemeris_time          ; convert UTC to ephemeris time (expressed as the number of ephemeris seconds past J2000)
    cspice_spkezr, Body, ephemeris_time, 'J2000', 'NONE', '0', state, light_time ;state is [x,y,z,vx,vy,vz] with respect to the solar system barycentre
    cspice_bodvrd, Body,'RADII', 3, Body_radius ; Find the simulated body's radius in Km
    Body_radius = Body_radius[0]
    state = state / Body_radius                 ; Convert state array's from km to units of planetary radii (used throughout this model)

    ; Load particle data like mass and cross-sections into a structure called particle_data, part of the model_shared common block
    LOAD_PARTICLE_DATA, Test_particle
    ; Load solar data in the region of interest to the emission line being modeled, part of the model_shared common block
    LOAD_LINE_DATA, Line

    maxtime = max(Time_range)*24.*3600. ; Change stop time from days to seconds
    mintime = min(Time_range)*24.*3600. ; Change start time from days to seconds

    FOR loop_number = 0, loop_times - 1 do begin ;How many times the program should loop to reduce statistical noise
      loop_number = fix(loop_number) ;convert the loop number to an integer

      loc = fltarr(10, Number_of_particles) ;Create a massive array of all particles
      ; loc(0,*) will be the x coords of all the particles in body radii
      ; loc(1,*) will be the y coords of all the particles in body radii
      ; loc(2,*) will be the z coords of all the particles in body radii
      ; loc(3,*) will be the bodycentric distance of all the particles in body radii
      ; loc(4,*) will be the fractional contant of the original packet (after they hit = 0, ionization (when included) should give a fraction of 1)
      ; loc(5,*) will be the x velocity of all the particles in body radii/s
      ; loc(6,*) will be the v velocity of all the particles in body radii/s
      ; loc(7,*) will be the z velocity of all the particles in body radii/s
      ; loc(8,*) will be the age = how long this particle has been tracked already (seconds)
      ; loc(9,*) will be the time between particle release and image taken in seconds (each packet of particles is released at a different time)

      Case n_elements(time_range) of
        1: loc[9,*] = replicate(mintime, Number_of_particles)                       ; A pulse of particles at point in time. Use for meteor impacts
        2: loc[9,*] = randomu(Seed, Number_of_particles)*(maxtime-mintime)+mintime  ; Random release times for some duration. Normal model runs use this.
      Endcase

      ; Make starting locations and velocities for particles
      print, 'Generating particle release states. . . '
      RELEASE_STATE, loc, speed_distribution, surface_distribution, speed

      ; The speed is normalized to 1, the velocity is randomly distributed in x,y,z directions
      ; Make starting locations for particles
      loc[0:2,*] = loc[0:2,*]*(Body_radius + exobase_height)    ; Scale the "release state" vectors to the exobase (km)
      loc(4,*) = 1.0                                            ; Packet content is full initially
      loc[5,*] = loc[5,*]*speed                                 ; Give them initial velocities in x; units are km/s, speed array is in km/s
      loc[6,*] = loc[6,*]*speed                                 ; Give them initial velocities in y; units are km/s, speed array is in km/s
      loc[7,*] = loc[7,*]*speed                                 ; Give them initial velocities in z; units are km/s, speed array is in km/s
      loc[8,*] = 0.0D                                           ; Particles start out at 0 seconds in age, they haven't aged yet

      ; Calculate the number of molecules/atoms that each test particle represents (test particles represent packets of atoms)
      if Maxtime eq mintime then duration = 1. else duration = maxtime-mintime ; duration over which the 'packets' of atoms are released [seconds]
      release_rate = number_of_particles / duration                            ; release rate of particles into the model in particles per second
      atoms_per_packet = Upward_flux_at_exobase / release_rate                 ; (atoms/s)/(packets/s), the average packet content loc[4,*] must be unity initially.

      ;    ; Calculate the Time-Dependent photo-ionization rate using the cross-section vs. wavelength UV fluxes come from SORCE and SEE
      ;      Print, 'Loading incident solar flux. . . '
      ;      Irrandiance, body, UTC, Time_range, directory, Flux, debug=debug   ;flux array is [wavelength in nm, solar photons s^(-1) cm^(-2)]
      ;      Ionize_lambda = 1239.84187 / particle_data.Ionization_potential    ;convert ionization potential from eV to nm wavelength
      ;      near = Min(Abs(float(Flux[0,*]) - Ionize_lambda), threshold)       ;only include wavelengths up to the ionization threshhold
      ;      flux = flux[*, 0:threshold]
      ;      interpolated_cross_sect = reform(INTERPOL(particle_data.photo_ionize_data[1,*], particle_data.photo_ionize_data[0,*], flux[0,*]))
      ;      ionizelife = (total(interpolated_cross_sect * flux[1,*])) ^ (-1.)  ;the e-folding lifetime in s
      ;      If keyword_set(debug) then begin                                   ;for direct comparison with Huebner et al., 1992
      ;        print, strcompress('Photo-ionization rate (s^-1) =' + string(ionizelife ^ (-1.)) + ', Lifetime = ' + string(ionizelife) + ' s')
      ;        Window, 0, Title = strcompress('Total ' + test_particle + ' Photolysis Rate = '+ string(ionizelife ^ (-1.)) + ' per second at ' + Body)
      ;        plot, flux[0,*]*10., interpolated_cross_sect*flux[1,*]/10., xrange = [0, Ionize_lambda*10.], /ylog, ystyle = 1., $
      ;          yrange = [1.e-14, 2.e-7], Xtitle = strcompress('Wavelength ' + cgSymbol("angstrom")), $
      ;          ytitle = strcompress('Photolysis Rate Coefficient s!U-1!N ' + cgSymbol("angstrom") + '!U-1!N'), psym=10, $
      ;          Color=cgColor('black'), Background=cgColor('white'), thick = 2., charthick = 1.6, charsize =1.6
      ;      endif

      Case test_particle of ; Force photoionization lifetime in seconds (Huebner & Mukherjee, 2015) Above calculation Irradiance.pro yeilds longer lifetimes!
        'Na': ionizelife  = 1./7.26e-6  ; Quiet Sun, Huebner & Mukherjee, 2015
        'Mg': ionizelife  = 1./6.49e-7  ; Huebner & Mukherjee, 2015
        'K' : ionizelife  = 1./2.70e-5  ; Huebner & Mukherjee, 2015
      endcase

      ; Integrate the equations of motion
      Print, 'Starting particle motion integration. . . '
      RK4_integrate_adaptive, loc, bounce, ionizelife, atoms_per_packet, timestep
      ; The loc array now contains final particle locations. Now to plot them taking into account shadowing and g-values

      ; Compute scattering rates for each packet
      final_time = loc[9,*] - loc[8,*]
      CSPICE_SPKEZR, body, ephemeris_time - REFORM(final_time), 'J2000', 'NONE', 'Sun', sun_state, ltime
      sun_part_pos = loc[0:2,*] + sun_state[0:2,*]  ;Calculate the vectors from the sun to the particles
      sun_part_vel = loc[5:7,*] + sun_state[3:5,*]
      r_sun = sqrt((sun_part_pos[0,*]^2. + sun_part_pos[1,*]^2. + sun_part_pos[2,*]^2.))  ;Particle distance from the Sun (units of km)
      Vrad = (sun_part_pos[0,*]*sun_part_vel[0,*] + $ ;radial velocity in KM/S WRT the Sun for all particles final states
        sun_part_pos[1,*]*sun_part_vel[1,*] + $
        sun_part_pos[2,*]*sun_part_vel[2,*]) / r_sun
      gvalue, Line_data.line, Vrad * 1000., r_sun / 149597871., Line_data.wavelength, Line_data.intensity, g

      ; Adjust scattering rates by the illuminated fraction of the solar disc
      airborn_packets = where(loc[4,*] ne 0., number_airborn, /NULL)
      penum = illumination(fltarr( number_airborn ), loc[0,airborn_packets], $
        loc[1,airborn_packets], loc[2,airborn_packets])
      g[airborn_packets] = g[airborn_packets] * penum ; Only sunlit packets emit

      ; Write the image for that loop.
      output_display, loc, g, atoms_per_packet, Image_type, loop_number, 1

      ; Build the images in both Rayleighs and Column Density by co-adding
      restore, strcompress(directory+'Model_Image_R'+string(loop_number)+'.sav')
      restore, strcompress(directory+'Model_Image_CD'+string(loop_number)+'.sav')
      Model_Cube_R[*,*,Loop_number] = Model_Image_R
      Model_Cube_CD[*,*,Loop_number] = Model_Image_CD
      if keyword_set(Above_Ecliptic) then begin
        restore, strcompress(directory+'Above_Ecliptic_Image_R'+string(loop_number)+'.sav')
        restore, strcompress(directory+'Above_Ecliptic_Image_CD'+string(loop_number)+'.sav')
        Above_Ecliptic_Cube_R[*,*,Loop_number] = Above_Ecliptic_Image_R
        Above_Ecliptic_Cube_CD[*,*,Loop_number] = Above_Ecliptic_Image_CD
      endif

      if N_elements(Time_range) gt 1 then save, loc, filename = strcompress(directory + Output_title + '_Loc_Array_'+string(loop_number)+'.sav') ; Save the big array for steady state release over some durations only
      print, 'Finished Particle Integration for Loop Number', Loop_number+1
    endfor

    ; Average (mean) the brightness and column density over the number of model runs in the above loop
    Model_Image_R  = mean(Model_Cube_R,  dimension = 3)
    Model_Image_CD = mean(Model_Cube_CD, dimension = 3)
    if keyword_set(Above_Ecliptic) then Above_Ecliptic_Image_R  = mean(Above_Ecliptic_Cube_R,  dimension = 3)
    if keyword_set(Above_Ecliptic) then Above_Ecliptic_Image_CD = mean(Above_Ecliptic_Cube_CD, dimension = 3)

    ; Make a FITS header and write in the settings above.
    MKHDR,    Header, fltarr(Output_Size_In_Pixels[0], Output_Size_In_Pixels[1])
    SXADDPAR, Header, 'UTC', UTC, ' UTC when image taken'
    SXADDPAR, Header, 'Upward_flux_at_exobase', Upward_flux_at_exobase, ' Upward flux at exobase [atoms/s]'
    SXADDPAR, Header, 'viewpoint', viewpoint, ' Observer location'
    SXADDPAR, Header, 'Surface_distribution', Surface_distribution, ' Surface Distribution'
    SXADDPAR, Header, 'FOV', FOV, ' Simulation Field of View [arcsec]'
    SXADDPAR, Header, 'Speed_distribution', Speed_distribution, ' Particle Speed Distribution'
    SXADDPAR, Header, 'Loop_times', Loop_times, ' Number of model loops'
    SXADDPAR, Header, 'N_particles', Number_of_particles, ' # of particles / loop'
    SXADDPAR, Header, 'Runtime', (systime(/seconds)-start_time)/3600., ' Simulation time [hours]'
    Case n_elements(time_range) of
      1: SXADDPAR, Header, 'time_range', time_range[0], ' Duration of particle integration [days]'
      2: ;SXADDPAR, Header, 'time_range', string(time_range[0]), ' Duration of particle integration [days]' need to write a string or a scalar here, what to do?
    Endcase
    MKHDR, ext_Header, fltarr(Output_Size_In_Pixels[0], Output_Size_In_Pixels[1]), /IMAGE ; make an image extention header

    ; Write an output FITS file with the mean brightness [extension 0], column density [extension 1] and header info about the input parameters simulated
    mwrfits, Model_Image_R,  strcompress(directory+Output_title+'.fit'), Header, /create, /Silent ; Write the brightness in Rayleighs
    mwrfits, Model_Image_CD, strcompress(directory+Output_title+'.fit'), ext_Header, /Silent      ; Append the column density per cm^-2 into FITS extension 1
    if keyword_set(Above_Ecliptic) then mwrfits, Above_Ecliptic_Image_R, strcompress(directory+Output_title+'_Above_Ecliptic.fit'), header, /create, /Silent
    if keyword_set(Above_Ecliptic) then mwrfits, Above_Ecliptic_Image_CD, strcompress(directory+Output_title+'_Above_Ecliptic.fit'), /Silent
    ; Example syntax for subsequent inspection:
    ; Brightness = mrdfits(strcompress(directory+Output_title+'.fit'), 0, header) & print, header
    ; Column_den = mrdfits(strcompress(directory+Output_title+'.fit'), 1, header)

    ; Plot the final loop-averaged results
    if keyword_set(Label_time) then Label_time = string(max(time_range)*24., format = '(F3.1)')
    output_display, loc, g, atoms_per_packet, Image_type, loop_number, 2, label_phase = label_phase, Label_time = Label_time

    ; Output_display can define spacecraft instrument pointing info, if there's any such information present
    ; add another fits binary table extention with this pointing info
    if keyword_set(Boresight_Pixel) then begin
      Pointing_info = {Pointing, Boresight_Pixel:Boresight_Pixel, Aperture_Corners:Aperture_Corners}
      mwrfits, Pointing_info, strcompress(directory+Output_title+'.fit') ; Append the pointing info into FITS extension 2
    endif
    ;      SXADDPAR, Header, 'BORE_X', Boresight_Pixel[0], 'Boresight X Nearest Pixel'
    ;      SXADDPAR, Header, 'BORE_Y', Boresight_Pixel[0], 'Boresight Y Nearest Pixel'
    ;    endif
    ;    if keyword_set(Aperture_Corners) then
    ;      SXADDPAR, Header, 'APER1X', Aperture_Corners[0,0], 'Aperture Corner 1 X Position'
    ;      SXADDPAR, Header, 'APER1Y', Aperture_Corners[1,0], 'Aperture Corner 1 Y Position'
    ;      SXADDPAR, Header, 'APER2X', Aperture_Corners[0,1], 'Aperture Corner 2 X Position'
    ;      SXADDPAR, Header, 'APER2Y', Aperture_Corners[1,1], 'Aperture Corner 2 Y Position'
    ;      SXADDPAR, Header, 'APER1X', Aperture_Corners[0,2], 'Aperture Corner 3 X Position'
    ;      SXADDPAR, Header, 'APER1Y', Aperture_Corners[1,2], 'Aperture Corner 3 Y Position'
    ;      SXADDPAR, Header, 'APER1X', Aperture_Corners[0,3], 'Aperture Corner 4 X Position'
    ;      SXADDPAR, Header, 'APER1Y', Aperture_Corners[1,3], 'Aperture Corner 4 Y Position'
    ;
    ;    endif
    ;    ;MODFITS, strcompress(directory+Output_title+'.fit'), 0, header ; Update header

    print,'Done, Number of model loops =', Loop_number
    print,'Execution Time =',(systime(/seconds)-start_time)/3600.,'   hours'
    return
  end