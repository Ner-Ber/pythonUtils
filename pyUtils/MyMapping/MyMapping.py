import numpy as np

def ps2ll(x,y,**kwargs):
    """
    THIS IS A MATLAB -> PYTHON TRANSLATION!!!!

    PS2LL transforms map coordinates to lat/lon data for a polar stereographic 
    system. This is a version of Andy Bliss' polarstereo_inv function, adapted 
    specifically for Antarctica. This function does NOT require the Mapping
    Toolbox. 
     
    #-- Syntax
     
     [lat,lon] = ps2ll(x,y) 
     [lat,lon] = ps2ll(x,y,'TrueLat',ReferenceLatitude) 
     [lat,lon] = ps2ll(x,y,'EarthRadius',RadiusInMeters) 
     [lat,lon] = ps2ll(x,y,'Eccentricity',EarthsMisshapenness) 
     [lat,lon] = ps2ll(x,y,'meridian',MeridianInDegrees) 
     
    #-- Description 
     
     [lat,lon] = ps2ll(x,y) transforms polar stereographic x,y coordinates (re: 
     71 S) to geographic lat/lon. Inputs x and y  can be scalar, vecotr, or
     matrices of equal size. 
     
     [lat,lon] = ps2ll(x,y,'TrueLat',ReferenceLatitude) secifies a reference
     latitude of true scale in degrees; also known as the standard parallel.
     Note that although Andy Bliss' polarstereo_inv function used -70 as a
     default, this function uses -71 as the default. NSIDC has been trying to
     standardize this, but take a close look at their reference latitudes for
     any data you put through this function--NSIDC sometimes uses 70 S, and
     sometimes uses 71 S. Again, the default in this function is -71. 
     
     [lat,lon] = ps2ll(x,y,'EarthRadius',RadiusInMeters) specifies Earth's
     radius in meters. Default is 6378137.0 m, WGS84.
     
     [lat,lon] = ps2ll(x,y,'Eccentricity',EarthsMisshapenness) specifies
     Earth's eccentricity or misshappenness.  Default values is 0.08181919. 
     
     [lat,lon] = ps2ll(x,y,'meridian',MeridianInDegrees) specifies the meridian in 
     degrees along the positive Y axis of the map. Default value is 0.
     
    #-- Snyder's example: Should return lat = -75 and lon = 150.
     
     x = -1540033.6;
     y = -560526.4;
     [lat,lon] = ps2ll(x,y,'EarthRadius',6378388.0,'eccentricity',0.0819919,'meridian',-100)
     
    #-- Author Info
     
     This function is a slightly adapted version of Andy Bliss' polarstereo_inv, 
     which can be found here: http://www.mathworks.com/matlabcentral/fileexchange/32907
     You can contact Andy at andybliss@gmail.com. 
     
     This function was tweaked a bit by Chad A. Greene of the University of Texas 
     at Austin's Institute for Geophysics (UTIG). Changes Chad made include removal
     of deg2rad and rad2deg to remove dependence on Mapping Toolbox, and a change to 
     71 degrees South as the reference latitude. 
     
    #-- Citing Antarctic Mapping Tools
     This function was developed for Antarctic Mapping Tools for Matlab (AMT). If AMT is useful for you,
     please cite our paper: 
     
     Greene, C. A., Gwyther, D. E., & Blankenship, D. D. Antarctic Mapping Tools for Matlab. 
     Computers & Geosciences. 104 (2017) pp.151-157. 
     http://dx.doi.org/10.1016/j.cageo.2016.08.003
     
     @article{amt,
       title={{Antarctic Mapping Tools for \textsc{Matlab}}},
       author={Greene, Chad A and Gwyther, David E and Blankenship, Donald D},
       journal={Computers \& Geosciences},
       year={2017},
       volume={104},
       pages={151--157},
       publisher={Elsevier}, 
       doi={10.1016/j.cageo.2016.08.003}, 
       url={http://www.sciencedirect.com/science/article/pii/S0098300416302163}
     }
       
    #-- Futher Reading
       
     Equations from: Map Projections - A Working manual - by J.P. Snyder. 1987 
     http://kartoweb.itc.nl/geometrics/Publications/Map%20Projections%20-%20A%20Working%20manual%20-%20by%20J.P.%20Snyder.pdf
     See the section on Polar Stereographic, with a south polar aspect and
     known phi_c not at the pole.
    
     WGS84 - radius: 6378137.0 eccentricity: 0.08181919
       in Matlab: axes2ecc(6378137.0, 6356752.3142)
     Hughes ellipsoid - radius: 6378.273 km eccentricity: 0.081816153
       Used for SSM/I  http://nsidc.org/data/polar_stereo/ps_grids.html
     International ellipsoid (following Snyder) - radius: 6378388.0 eccentricity: 0.0819919 
     
     See also: LL2PS, PROJINV, PROJFWD, MINVTRAN, MFWDTRAN, and ROTATEM.

    Args:
        x ([type]): [description]
        y ([type]): [description]

    Returns:
        [type]: [description]
    """

    #---  Set defaults: 
    phi_c_def = -71;   # standard parallel - this is different from Andy Bliss' function, which uses -70! 
    a_def = 6378137.0; # radius of ellipsoid, WGS84
    e_def = 0.08181919;# eccentricity, WGS84
    lambda_0_def = 0;  # meridian along positive Y axis

    kwargs.setdefault('phi_c',  phi_c_def)
    kwargs.setdefault('a',  a_def)
    kwargs.setdefault('e',  e_def)
    kwargs.setdefault('lambda_0',  lambda_0_def)

    phi_c = kwargs['phi_c']
    a = kwargs['a']
    e = kwargs['e']
    lambda_0 = kwargs['lambda_0']


    #--- Convert to radians and switch sign because this function is southern-hemisphere-specific: 
    phi_c = -phi_c*np.pi/180;
    lambda_0 = -lambda_0*np.pi/180;
    x=-x;
    y=-y;

    #this is not commented very well. See Snyder for details.
    t_c=np.tan(np.pi/4 - phi_c/2)/((1-e*np.sin(phi_c))/(1+e*np.sin(phi_c)))**(e/2)
    m_c=np.cos(phi_c)/np.sqrt(1-e**2*(np.sin(phi_c))**2);
    rho=np.sqrt(x**2+y**2); 
    t=rho*t_c/(a*m_c);

    #find phi with a series instead of iterating.
    chi=np.pi/2 - 2 * np.arctan(t);
    lat=chi+(e**2/2 + 5*e**4/24 + e**6/12 + 13*e**8/360)*np.sin(2*chi) \
        + (7*e**4/48 + 29*e**6/240 + 811*e**8/11520)*np.sin(4*chi) \
        + (7*e**6/120+81*e**8/1120)*np.sin(6*chi) \
        + (4279*e**8/161280)*np.sin(8*chi);

    lon= lambda_0 + np.arctan2(x,-y);

    #correct the signs and phasing
    lat=-lat;
    lon=-lon;
    lon=np.mod(lon+np.pi,2*np.pi)-np.pi; #want longitude in the range -pi to pi

    #convert back to degrees
    lat=lat*180/np.pi;
    lon=lon*180/np.pi;


    return lat,lon


if __name__=="__main__":
    x = -265.669
    y = -577.157
    lat, lon = ps2ll(x,y)

    pass