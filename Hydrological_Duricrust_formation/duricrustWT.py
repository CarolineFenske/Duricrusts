import scipy.fftpack as fft
import numba as nb
import xsimlab as xs
import numpy as np

@nb.jit(nopython=True, cache=True)
def TVD_FVM(phi, v, s, dx, dt, bc="fixed"):
    """
    function to solve the advection equation
    dphi/dt = -v dphi/dx + s
    using a Fux Limiting Explicit Total Variation Diminishing Finite Volume Method (TVD-FVM)
    as described in

    in input:
    - phi, array containing the field to be advected
    - v, the advection velocity (scalar)
    - S, the source term (array of dimension like phi)
    - dx, uniform spatial discretization
    - dt, time step
    - bc, boundary condition type on rhs boundary (phi[-1])
            ('fixed' or 'no_flux')

    in output:
    - phi, the advected field

    Note that the function compute the number of time steps
    necessary to fulfill the CFL condition

    """

    # nx is length of phi or number of nodes where the solution is computed
    nx = len(phi)
    # CFL is CFL condition (must be <1 for stability)
    CFL = v / dx * dt

    # we compute the number of time steps necessary for the solution to be stable
    nstep = 1
    if CFL > 1:
        nstep = int(CFL) + 1

    # we compute the resulting time step and corresponding advection term
    dti = dt / nstep
    F = v / dx * dti

    # we perform nstep intermediary time steps
    for step in range(nstep):
        # phi0 is a copy of phi at time t
        phi0 = phi.copy()

        rp = np.zeros_like(phi)
        den = phi0[1:-1] - phi0[2:]
        num = np.where(den != 0, phi0[:-2] - phi0[1:-1], 0)
        den = np.where(den != 0, den, 1)
        rp[2:] = num / den
        phip = (rp + np.abs(rp)) / (1 + np.abs(rp))

        rm = np.zeros_like(phi)
        rm[1:-1] = rp[2:]
        phim = (rm + np.abs(rm)) / (1 + np.abs(rm))

        # we update phi
        phi[1:-1] = (
            phi0[1:-1]
            + dti * s[1:-1]
            - (
                F * (phi0[1:-1] - phi0[:-2])
                + phim[1:-1] * F * (1 - F) / 2 * (phi0[2:] - phi0[1:-1])
                - phip[1:-1] * F * (1 - F) / 2 * (phi0[1:-1] - phi0[:-2])
            )
        )

        # we apply the rhs boundary condition (no gradient in phi)
        if bc == "no_flux":
            phi[-1] = phi[-2]

    return phi

@nb.jit(nopython=True, cache=True)
def integrate(x):
    """
    integrate:
    ---------

    Function to integrate variable x by simple summation:
    int_i = sum_(j = 1 to i) of x_j

    in input:
    x: array to integrate

    in output:
    integ: result of the integration

    """

    integ = x.copy()
    for i in range(len(integ) - 1, 0, -1):
        integ[i - 1] = integ[i - 1] + integ[i]

    return integ


@nb.jit(nopython=True, cache=True)
def tridag(a, b, c, r):
    """
    tridag:
    ------

    Function to solve a tri-diagonal system of n equations

    in input:
    a is lower diagonal (length n but first value is not used)
    b is diagonal (length n)
    c is upper diagonal (length n but last value is not used)
    r is rhigh hand side vector (length n)

    in output:
    tridag returns an array of size n containing the solution
    to the system of equations
    """

    n = len(r)
    res = np.empty(n)
    gam = np.empty(n)

    if b[0] == 0:
        raise RuntimeError("diagonal contains nil element in tridag")

    bet = b[0]
    res[0] = r[0] / bet
    for j in range(1, n):  # in Fortran the range is 2,n
        gam[j] = c[j - 1] / bet
        bet = b[j] - a[j] * gam[j]
        if bet == 0:
            raise RuntimeError("diagonal contains nil element in tridag")
        res[j] = (r[j] - a[j] * res[j - 1]) / bet

    for j in range(n - 2, -1, -1):  # in Fortran the range is n-1,1,-1
        res[j] = res[j] - gam[j + 1] * res[j + 1]

    return res


@nb.jit(nopython=True, cache=True)
def table(B, z, accum, dx, K):
    """
    table:
    -----

    Function to compute the geometry of the water table
    using the second-order accurate finite difference scheme
    described in Braun et al (2015)

    in input:
    B: the thickness of the regolith
    z: the surface topography
    accum: cumulative infiltration rate
    dx: distance between two nodes
    K: hydraulic conductivity


    in output:
    H: water table height

    """

    H = z.copy()
    for i in range(1, len(H)):
        b = -z[i] - z[i - 1] + B[i] + B[i - 1]
        c = (z[i] + z[i - 1] - B[i] - B[i - 1] - H[i - 1]) * H[i - 1] - 2 * dx * accum[
            i
        ] / K
        H[i] = (-b + np.sqrt(b ** 2 - 4 * c)) / 2
        if H[i] > z[i]:
            H[i] = z[i]
        if H[i] < z[i] - B[i]:
            H[i] = z[i] - B[i]
            
    velo = z.copy()
    velo[1:-1] = K * (H[2:] - H[:-2]) / dx / 2
    velo[0] = K * (H[1] - H[0]) / dx
    velo[-1] = K * (H[-1] - H[-2]) / dx
    
    return H, velo


@nb.jit(nopython=True, cache=True)
def linear_interpolate(xx, x, y, left, right):
    """
    linear_interpolate:
    ------------------

    Function to linearly interpolate a function y(x) defined as
    a set of pairs (x,y), where the x values are given in increasing
    order. The interpolation is performed at for a set of xx points.
    left and right are the values used for the interpolation
    outside of the range [x[0],x[-1]]. If left and/or right are
    not defined, it extrapolates the values to the left to y[0]
    and to the right to y[-1]

    in input:
    xx: the x-locations of the points where the function is to be
        interpolated
    x: the x-coordinates used to define the function
    y: the values y(x) used to define the function
    left: values used to the left of x[0]
    right: value used to the right of x[-1]

    in output:
    returns the interpolated values at points of x-coordinate xx

    """
    
    n = len(x)
    nn = len(xx)
    i = 0
    yy = np.ones_like(xx) * left
    for ii in range(nn):
        xxx = xx[ii]
        while i < n and xxx > x[i]:
            i = i + 1
        if i == n:
            yy[ii:] = right
            return yy
        r = 0.5
        if x[i] != x[i - 1]:
            r = (xxx - x[i - 1]) / (x[i] - x[i - 1])
        if r <= 1:
            yy[ii] = y[i - 1] + r * (y[i] - y[i - 1])

    return yy


@nb.jit(nopython=True, cache=True)
def hardeningWT(hardness, wtb, B, dB, dtopo, tau, Precip, lamda, dt):
    """
    hardening:
    ---------

    Function to update the value of a hardening factor (hardness)
    along a vertical regolith profile of thickness B
    by a process of efficiency proportional to the distance to
    water table located at a position wth from the base of the
    profile
    Note that there is no gradient in hardening along the left
    hand-side boundary (base level)

    in input:
    hardness: initial value of the hardening factor (shape is (ny,nx))
    wtb: water table height (measured from base of regolith layer, length nx)
    B: regolith layer thickness (length nx)
    dB: incremental change in regolith thickness (length nx)
    dtopo: incremental change in surface topography (length nx)
    tau: characteristic time scale for duricrust formation
    Precip: precipitation rate
    lamda: Distance over which the water table is beating (seasonally)

    in output:
    returns the updated value of the hardening factor

    """

    advection = "True"
    Pref = 5 # reference precipitation rate

    ny, nx = hardness.shape
    for loc in range(1, len(B)):
        y = np.linspace(0, B[loc], ny)
        if advection:
            yp = np.linspace(0, B[loc] - dB[loc] - dtopo[loc], ny)
        else:
            yp = np.linspace(dB[loc], B[loc] - dtopo[loc], ny)
        if (B[loc] - dB[loc] - dtopo[loc]) > 0:
            hardness[:, loc] = linear_interpolate(y, yp, hardness[:, loc], 1., 1.)
        dy = B[loc] / (ny - 1)
        if dy > 0:
            if advection:
                v = dB[loc] / dt
            else:
                v = 0
            hardness[:, loc] = TVD_FVM(
                hardness[:, loc],
                v,
                -hardness[:, loc] / tau * Precip/Pref * np.exp(-((y - wtb[loc]) ** 2) / lamda ** 2),
                dy,
                dt,
                "no_flux",
            )

    return hardness

@nb.jit(nopython=True, cache=True)
def ageing(age_regolith, age_duricrust, wtb, B, dB, dtopo, lamda, dt, time):
    """
    aging:
    ---------

    Function to update the value of the regolith and duricrust ages
    along a vertical regolith profile of thickness B
    
    in input:
    age_regolith: initial value of the regolith age
    age_duricrust: initial value of the duricrust age
    B: regolith layer thickness (length nx)
    dB: incremental change in regolith thickness (length nx)
    dtopo: incremental change in surface topography (length nx)

    in output:
    returns the updated value of the hardening factor

    """

    ny, nx = age_regolith.shape
    for loc in range(1, len(B)):
        if B[loc]>0:
            y = np.linspace(0, B[loc], ny)
            yp = np.linspace(dB[loc], B[loc] - dtopo[loc], ny)
            age_regolith[0,loc] = time
            for j in range(ny):
                if (y[j]-wtb[loc]+lamda)*(y[j]-wtb[loc]-lamda)<0: age_duricrust[j,loc] = time
            if (B[loc] - dB[loc] - dtopo[loc])>0:
                age_regolith[:, loc] = linear_interpolate(y, yp, age_regolith[:, loc], time, time)
                age_duricrust[:, loc] = linear_interpolate(y, yp, age_duricrust[:, loc], time, time)
            age_regolith[-1,loc] = age_regolith[-2,loc]
            age_duricrust[-1,loc] = age_duricrust[-2,loc]
            
    return age_regolith, age_duricrust

@nb.jit(nopython=True, cache=True)
def erosionDiffusion(topography, Kd, dt, dx):
    """
    erosion:
    -------

    Function to update the value of the surface topography known
    at nx equidistant points using a simple linear diffusion law
    based on an implicit algorithm

    in input:
    topography: intial surface topography (length nx)
    Kd: transport coefficient (length nx)
    dt: time step/increment
    dx: spatial increment

    in output:
    returns the updated value of the surface topography

    """

    diag = np.empty_like(topography)
    sup = np.empty_like(topography)
    inf = np.empty_like(topography)

    fact = dt / dx ** 2 / 2
    diag[1:-1] = (Kd[2:] + 2 * Kd[1:-1] + Kd[:-2]) * fact + 1
    sup[:-1] = -(Kd[:-1] + Kd[1:]) * fact
    inf[1:] = -(Kd[1:] + Kd[:-1]) * fact
    rhs = topography.copy()

    diag[0] = 1
    sup[0] = 0
    diag[-1] = 1
    inf[-1] = -1
    rhs[-1] = 0

    return tridag(inf, diag, sup, rhs)

#-----------------------------------------------------------

@xs.process
class Mesh:
    """
    Process that creates the spatial discretization of the model from its geometry
    """
    nx = xs.variable(description='number of nodes in the horizontal x-direction')
    L = xs.variable(description='horizontal length of the model',
                   attrs={'units': 'm'})
    dx = xs.variable(intent='out',
                    description='horizontal spatial step',
                    attrs={'units': 'm'})
    x = xs.index(dims='x',
                description='horizontal x-coordinate',
                attrs={'units': 'm'})
    ny = xs.variable(description='number of points in the vertical y-direction')
    y = xs.index(dims='y',
                description='vertical y-coordinates',
                attrs={'units': 'm'})
    
    def initialize(self):
        self.dx = self.L/(self.nx - 1)
        self.x = np.linspace(0, self.L, self.nx)
        self.y = np.linspace(0, 1, self.ny)
        
@xs.process
class Precipitation:
    """
    Process to accumulate precipitation into runoff
    """
    rate = xs.variable(dims=[(),'x'],
                      description='precipitation rate',
                      attrs={'units': 'm/yr'})
    accum = xs.variable(dims='x', intent='out',
                       description='integrated precipitation rate',
                       attrs={'units': 'm^2/yr'})
    rate_variable = xs.variable(dims='x', intent='out',
                               description='spatially variable precipitation rate',
                               attrs={'units': 'm/yr'})
    nx = xs.foreign(Mesh, 'nx')
    dx = xs.foreign(Mesh, 'dx')
    
    def run_step(self):
        self.rate_variable = np.broadcast_to(self.rate, self.nx)
        rate = self.rate_variable * self.dx
        self.accum = integrate(rate)

@xs.process
class Topography:
    """
    Process to update topography from uplift and erosion over time step
    """
    elevation = xs.variable(dims='x', intent='inout',
                           description='surface topography elevation',
                           attrs={'units': 'm'})
    dtopo_up_tot = xs.variable(dims='x', intent='inout',
                              description='total uplift over time step',
                              attrs={'units': 'm'})
    dtopo_down_tot = xs.variable(dims='x', intent='inout',
                                  description='total erosion over time step',
                                  attrs={'units': 'm'})
    dtopo_up = xs.group('dtopo_up')
    dtopo_down = xs.group('dtopo_down')
    nx = xs.foreign(Mesh, 'nx')

    def run_step(self):
        self.dtopo_down_tot = sum((dtopo for dtopo in self.dtopo_down))
        self.dtopo_up_tot = sum((dtopo for dtopo in self.dtopo_up))
        
    def finalize_step(self):
        self.elevation[1:] = self.elevation[1:] + self.dtopo_down_tot[1:] + self.dtopo_up_tot[1:]

@xs.process
class Uplift:
    """
    Process to compute surface uplift by tectonic processes
    """
    dtopo = xs.variable(dims='x', intent='out', groups='dtopo_up',
                       description='uplift over time step',
                       attrs={'units': 'm'})
    rate = xs.variable(dims=[(),'x'],
                      description='uplift rate',
                      attrs={'units': 'm/yr'})
    nx = xs.foreign(Mesh, 'nx')
    
    def initialize(self):
        self.dtopo = np.zeros(self.nx)
    
    @xs.runtime(args="step_delta")
    def run_step(self, dt):
        rate = np.broadcast_to(self.rate, self.nx)
        self.dtopo[1:] = rate[1:] * dt



@xs.process
class Diffusion:
    """
    Process to compute erosion due to hillslope processes represented by linear diffusion
    """
    dtopo = xs.variable(dims='x', intent='out', groups='dtopo_down',
                       description='diffusion erosion over time step',
                       attrs={'units': 'm'})
    Kd = xs.variable(dims=[(),'x'],
                    description='diffusivity (transport coefficient)',
                    attrs={'units': 'm^2/yr'})
    h = xs.foreign(Topography, 'elevation')
    dx = xs.foreign(Mesh, 'dx')
    precip = xs.foreign(Precipitation, 'rate_variable')
    nx = xs.foreign(Mesh, 'nx')
    
    @xs.runtime(args="step_delta")
    def run_step(self, dt):
        Kd = np.broadcast_to(self.Kd, self.nx)
        self.dtopo = erosionDiffusion(self.h, Kd, dt, self.dx) - self.h

        
@xs.process
class Regolith:
    """
    Process to compute the geometry of the regolith
    and the position of the water table
    """
    weathering_front = xs.variable(dims='x', intent='out',
                                  description='weathering front height',
                                  attrs={'units': 'm'})
    water_table = xs.variable(dims='x', intent='out',
                             description='water table height',
                             attrs={'units': 'm'})
    WTB = xs.variable(dims='x', intent='out',
                     description='water table height wrt base of regolith',
                     attrs={'units': 'm'})
    K = xs.variable(description='hydraulic conductivity',
                   attrs={'units': 'm/yr'})
    F = xs.variable(description='ratio of weathering front velocity over fluid velocity')
    
    Omega = xs.on_demand(description='Dimensionless number that controls regolith thickness (no regolith when <1)')
    
    Gamma = xs.on_demand(description='Dimensionless numbner that controls relationship to topography (thickest at the base when Gamma<Omega^2/(Omega-1))')
    
    dB = xs.variable(dims='x', intent='out',
                    description='increment in front height over time step',
                    attrs={'units': 'm'})
    velo = xs.variable(dims='x', intent='out',
                      description='fluid velocity',
                      attrs={'units': 'm/yr'})
    thickness = xs.variable(dims='x', intent='inout',
                           description='regolith thickness',
                           attrs={'units': 'm'})
    erate = xs.variable(dims = 'x', intent = 'out')
    dtopo = xs.foreign(Topography, 'dtopo_down_tot')
    accum = xs.foreign(Precipitation, 'accum')
    dx = xs.foreign(Mesh, 'dx')
    nx = xs.foreign(Mesh, 'nx')
    ny = xs.foreign(Mesh, 'ny')
    h = xs.foreign(Topography, 'elevation')
    precip = xs.foreign(Precipitation, 'rate_variable')
    
    def initialize(self):
        self.water_table = self.h.copy()
        self.WTB = np.zeros_like(self.h)

    @xs.runtime(args="step_delta")
    def run_step(self, dt):
        self.water_table, self.velo = table(
            self.thickness, self.h, self.accum, self.dx, self.K)
        self.dB = self.thickness.copy()
        self.thickness[1:] = (
            self.thickness[1:]
            + dt
            * self.F
            * self.K
            * (self.water_table[1:] - self.water_table[:-1])
            / self.dx
            + self.dtopo[1:]
        )
        # apply boundary condition at base level
        self.thickness[0] = (2 * self.thickness[1] - self.thickness[2])  
        # adjust regolith thickness to avoid negative values
        self.thickness = np.where(self.thickness > 0, self.thickness, 0)  
        # compute position of weathering front
        self.weathering_front = self.h - self.thickness
        # store incremental regolith thickness change
        self.dB = (self.thickness - self.dB - self.dtopo)  
        # again check that no negative value is included
        self.dB = np.where(self.dB > 0, self.dB, 0)          
        # compute water table height measured from base of regolith
        self.WTB = self.thickness - (self.h - self.water_table)
        self.erate = -self.dtopo/dt

    @Omega.compute
    def _Omega(self):
        slope = np.gradient(self.h, self.dx)
        slope_mean = np.mean(slope)
        Omega = self.F*self.K*slope_mean/np.mean(self.erate) 
        return Omega

    @Gamma.compute
    def _Gamma(self):
        slope = np.gradient(self.h, self.dx)
        slope_mean = np.mean(slope)
        Gamma = self.K*slope_mean**2/np.mean(self.precip) 
        return Gamma
    
        
@xs.process
class Duricrust:
    """
    Generic process to compute the formation of a duricrust inside the regolith
    """

    hardness = xs.variable(dims=('y','x'), intent='inout',
                          description='relative hardness factor')   
    front = xs.foreign(Regolith, 'weathering_front')
    table = xs.foreign(Regolith, 'WTB')
    dtopo = xs.foreign(Topography, 'dtopo_down_tot')
    h = xs.foreign(Topography, 'elevation')
    nx = xs.foreign(Mesh, 'nx')
    ny = xs.foreign(Mesh, 'ny')
    thickness = xs.foreign(Regolith, 'thickness')
    dB = xs.foreign(Regolith, 'dB')
    precip = xs.foreign(Precipitation, 'rate')
                
@xs.process
class DuricrustWaterTable(Duricrust):
    """
    Duricrust process by beating of the water table
    """
    tau = xs.variable(description='characteristic time scale for duricrust formation',
                     attrs={'units': 'yr'})
    lamda = xs.variable(description='water table beating height',
                       attrs={'units': 'm'})

    def initialize(self):
        self.y = np.linspace(0, 1, self.ny)

    @xs.runtime(args=("step_delta","step_start"))
    def run_step(self, dt, time):
        self.hardness = hardeningWT(
            self.hardness,
            self.table,
            self.thickness,
            self.dB,
            self.dtopo,
            self.tau,
            self.precip,
            self.lamda,
            dt,
        )  # compute hardening

@xs.process
class HardenDiffusion:
    """
    Process to modify the diffusion transport coefficient as a function of duricrust formation
    """
    hardness = xs.foreign(Duricrust, 'hardness')
    Kd = xs.variable(description='reference diffusivity',
                    attrs={'units': 'm^2/yr'})
    Kdv = xs.foreign(Diffusion, 'Kd', intent='out')

    def initialize(self):
        self.Kdv = self.hardness[-1, :] * self.Kd

    def run_step(self):
        self.Kdv = (
            self.hardness[-1, :] * self.Kd
        )  # adjust rock transport coefficient by hardening factor
        
@xs.process
class Age_WTduricrust:
    """
    Age process for the regolith and for duricrusts
    """
    age_regolith = xs.variable(intent="inout", dims=("y", "x"),
                                description="Regolith age",
                                attrs={"units": "yr"})    
    age_duricrust = xs.variable(intent="inout", dims=("y", "x"),
                                description="Duricrust age",
                               attrs={"units": "yr"}) 
    table = xs.foreign(Regolith, 'WTB')
    thickness = xs.foreign(Regolith, 'thickness')
    dB = xs.foreign(Regolith, 'dB')
    dtopo = xs.foreign(Topography, 'dtopo_down_tot')
    lamda = xs.foreign(DuricrustWaterTable, 'lamda')

    @xs.runtime(args=("step_delta","step_start"))
    def run_step(self, dt, time):
        self.age_regolith, self.age_duricrust = ageing(
            self.age_regolith,
            self.age_duricrust,
            self.table,
            self.thickness,
            self.dB,
            self.dtopo,
            self.lamda,
            dt,
            time,
        )  # compute age
               

@xs.process
class InitTopography:
    """
    Process to initialize the topography
    """
    slope = xs.variable(description='initial topographic slope')
    elevation = xs.foreign(Topography, 'elevation', intent='out')
    nx = xs.foreign(Mesh, 'nx')
    L = xs.foreign(Mesh, 'L')
    
    def initialize(self):
        self.elevation = np.linspace(0, self.slope*self.L, self.nx)
        
@xs.process
class InitWTRegolith:
    """
    Process to initialize regolith thickness
    """
    thickness = xs.foreign(Regolith, 'thickness', intent='out')
    age_regolith = xs.foreign(Age_WTduricrust, 'age_regolith', intent='out')
    nx = xs.foreign(Mesh, 'nx')
    ny = xs.foreign(Mesh, 'ny')
    
    def initialize(self):
        self.thickness = np.ones(self.nx)
        self.age_regolith = np.zeros((self.ny, self.nx))
        
@xs.process
class InitWTDuricrust:
    """
    Process to initialize the duricrust hardness
    """
    hardness = xs.foreign(Duricrust, 'hardness', intent='out')
    age_duricrust = xs.foreign(Age_WTduricrust, 'age_duricrust', intent='out')
    nx = xs.foreign(Mesh, 'nx')
    ny = xs.foreign(Mesh, 'ny')
    
    def initialize(self):
        self.hardness = np.ones((self.ny, self.nx))
        self.age_duricrust = np.zeros((self.ny, self.nx))


@xs.process
class InitDummy:
    """
    Dummy process to initialize a variety of group variables
    """
    dtopo_up_tot = xs.foreign(Topography, 'dtopo_up_tot', intent='out')
    dtopo_down_tot = xs.foreign(Topography, 'dtopo_down_tot', intent='out')
    nx = xs.foreign(Mesh, 'nx')
#    ny = xs.foreign(Mesh, 'ny')
    
    def initialize(self):
        self.dtopo_up_tot = np.zeros(self.nx)
        self.dtopo_down_tot = np.zeros(self.nx)

#--------------------------------------------------------------------------

water_table_model = xs.Model({'mesh': Mesh,
                  'precip': Precipitation,
                  'topography': Topography,
                  'diffusion': Diffusion,
                  'uplift': Uplift,
                  'regolith': Regolith,
                  'duricrust': DuricrustWaterTable,
                  'age': Age_WTduricrust,
                  'harden_diffusion': HardenDiffusion,
                  'init_topography': InitTopography,
                  'init_regolith': InitWTRegolith,
                  'init_duricrust': InitWTDuricrust,
                  'init_dummy': InitDummy})