import QuantLib as ql
from math import sqrt

class FXSmileFit:
    def __init__(self, spot, term, dom_r, for_r, quote,
                delta_type=ql.DeltaVolQuote.PaFwd, atm_type=ql.DeltaVolQuote.AtmDeltaNeutral,
                calendar=ql.UnitedStates(), day_count=ql.Actual365Fixed(), ref_date=ql.Date().todaysDate()):
        self.spot = spot
        self.term = term
        self.dom_r = dom_r
        self.for_r = for_r
        self.quote = quote
        self.delta_type = delta_type
        self.atm_type = atm_type
        self.calendar = calendar
        self.day_count = day_count
        self.ref_date = ref_date
        self.ah_surface = None
    
    def fit(self):
        vol_atm = self.quote[0] / 100
        vol_put_10d = (self.quote[0] + self.quote[4] - self.quote[3] / 2) / 100
        vol_put_25d = (self.quote[0] + self.quote[2] - self.quote[1] / 2) / 100
        vol_call_25d = (self.quote[0] + self.quote[2] + self.quote[1] / 2) / 100
        vol_call_10d = (self.quote[0] + self.quote[4] + self.quote[3] / 2) / 100

        quote_spot = ql.SimpleQuote(self.spot)
        quote_vol_atm = ql.SimpleQuote(vol_atm)
        quote_vol_put_10d = ql.SimpleQuote(vol_put_10d)
        quote_vol_put_25d = ql.SimpleQuote(vol_put_25d)
        quote_vol_call_25d = ql.SimpleQuote(vol_call_25d)
        quote_vol_call_10d = ql.SimpleQuote(vol_call_10d)

        ql_atm_vol = ql.DeltaVolQuote(ql.QuoteHandle(quote_vol_atm), self.delta_type, self.term, self.atm_type)
        ql_10d_put_vol = ql.DeltaVolQuote(-0.1, ql.QuoteHandle(quote_vol_put_10d), self.term, self.delta_type)
        ql_25d_put_vol = ql.DeltaVolQuote(-0.25, ql.QuoteHandle(quote_vol_put_25d), self.term, self.delta_type)
        ql_25d_call_vol = ql.DeltaVolQuote(0.25, ql.QuoteHandle(quote_vol_call_25d), self.term, self.delta_type)
        ql_10d_call_vol = ql.DeltaVolQuote(0.1, ql.QuoteHandle(quote_vol_call_10d), self.term, self.delta_type)

        rd_ts = ql.YieldTermStructureHandle(ql.FlatForward(self.ref_date, self.dom_r, self.day_count))
        rf_ts = ql.YieldTermStructureHandle(ql.FlatForward(self.ref_date, self.for_r, self.day_count))

        d_df, f_df = rd_ts.discount(self.term), rf_ts.discount(self.term)

        calc_atm = ql.BlackDeltaCalculator(ql.Option.Call, self.delta_type, self.spot, d_df, f_df, sqrt(self.term) * vol_atm)
        calc_10d_put = ql.BlackDeltaCalculator(ql.Option.Put, self.delta_type, self.spot, d_df, f_df, sqrt(self.term) * vol_put_10d)
        calc_25d_put = ql.BlackDeltaCalculator(ql.Option.Put, self.delta_type, self.spot, d_df, f_df, sqrt(self.term) * vol_put_25d)
        calc_25d_call = ql.BlackDeltaCalculator(ql.Option.Call, self.delta_type, self.spot, d_df, f_df, sqrt(self.term) * vol_call_25d)
        calc_10d_call = ql.BlackDeltaCalculator(ql.Option.Call, self.delta_type, self.spot, d_df, f_df, sqrt(self.term) * vol_call_10d)

        vols = [vol_put_10d, vol_put_25d, vol_atm, vol_call_25d, vol_call_10d]
        strikes = [
            calc_10d_put.strikeFromDelta(-0.1),
            calc_25d_put.strikeFromDelta(-0.25),
            calc_atm.atmStrike(self.atm_type),
            calc_25d_call.strikeFromDelta(0.25),
            calc_10d_call.strikeFromDelta(0.1)]

        calibration_set = ql.CalibrationSet()
        for strike, vol in zip(strikes, vols):
            payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
            exercise = ql.EuropeanExercise(self.ref_date + ql.Period(self.term, ql.Years))
            option = ql.VanillaOption(payoff, exercise)
            calibration_set.push_back((option, ql.SimpleQuote(vol)))
        ah_interpolation = ql.AndreasenHugeVolatilityInterpl(calibration_set, ql.QuoteHandle(quote_spot), rd_ts, rf_ts)
        
        self.ah_surface = ql.AndreasenHugeVolatilityAdapter(ah_interpolation)

    def get_vol_from_stike(self, strike):
        if self.ah_surface is None:
            self.fit()
        return self.ah_surface.blackVol(self.term, strike)