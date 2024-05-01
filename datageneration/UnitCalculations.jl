import Pkg; Pkg.add("Unitful")
import Pkg; Pkg.add("UnitfulAstro")

using Unitful
using UnitfulAstro

##### A SCALE CALCULATION ######################

GM_sun = (1u"GMsun" |> upreferred).val
GM_sun_length = ((1*u"GMsun" / (1*u"c"^2)) |> upreferred).val
light_second = ((1u"c" * 1u"s") |> upreferred).val
kilo_parsec = ((1000*u"pc") |> upreferred).val

c = (1u"c" |> upreferred).val
d_scale = kilo_parsec

A_scale = ( (GM_sun_length^5 / light_second^2)^(1/3) ) / (kilo_parsec)
@show A_scale = A_scale       #1.385328279341387e-20

#### F SCALE CALCULATION #######################

r_scale = 1e8 #m
GM_sun_units = ((1*u"GMsun") |> upreferred )
r_scale_units = 1e8*u"m"


f_scale = ( GM_sun_units^(1/2) / ((r_scale_units)^(3/2)) |> upreferred)
@show f_scale = f_scale
f_scale = f_scale.val
f_scale        # = 0.011520088541326409