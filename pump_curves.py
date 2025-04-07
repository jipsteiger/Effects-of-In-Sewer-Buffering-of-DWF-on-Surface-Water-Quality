from scipy.interpolate import interp1d


class EsPumpCurve:
    depth = [0, 0.138, 0.36, 1.0048, 3]
    flow = [0, 0.5, 1.42, 3.888, 3.888]
    interpolated_curve = interp1d(depth, flow, kind="linear", fill_value="extrapolate")


class RzPumpCurve:
    depth = [0, 1.16, 2.3208, 2.3210, 8.703, 8.710, 23.208, 29.01]
    flow = [0, 0.3254, 0.857, 1.208, 1.208, 2.5, 2.5, 4.7222]
    interpolated_curve = interp1d(depth, flow, kind="linear", fill_value="extrapolate")
