library(readr)
load("~/Downloads/super_data.Rdata")
Super <- apra.sr.full %>% group_by (Year) %>%
  mutate (Funds_to_market = sum(Total_contributions_, na.rm=TRUE),
          Total_payments= sum(Total_benefit_payments_, na.rm=T),
          mr = mean (Rate_of_return, na.rm=TRUE),
          number_of_funds = n(),
          TotalFUM = sum(last(Total_assets_at_end_of_period_)),
          TotalCont = sum(last(Total_contributions_)),
          TotalWithd = sum(last(Total_benefit_payments_)),
          TotalFUMstart = sum(first(Total_assets_at_end_of_period_)),
          TotalContstart = sum(first(Total_contributions_)),
          TotalWithdstart = sum(first(Total_benefit_payments_))) %>%
  group_by (Fund_name) %>%
  mutate (ms = Total_contributions_+Net_rollovers_, 
          lag_mr = lag(mr),
          performance = Rate_of_return - mr,
          lag_perf = lag(performance),
          lag2_perf = lag(lag_perf),
          lag3_perf = lag(lag2_perf),
          perf_3year = (lag_perf+lag2_perf+lag3_perf)/3,
          FUM = Total_assets_at_end_of_period_,
          over_50 = ((X50_59_female + X50_59_male+X60_65_female+X60_65_male+
                        X66_female+X66_male)/Number_of_members)*100,
          female = ((X_35_female+X35_49_female+X50_59_female+
                       X60_65_female+X66_female)/Number_of_members)*100) %>%
  dplyr::select(Fund_name, Year, ms, lag_mr, performance, lag_perf, FUM, over_50, female, Average.fees, Fund_type, Cost.ratio) %>%
  ungroup %>% group_by(Year) %>%
  mutate(ms = ms/sum(ms, na.rm = T),
         ms = 100*ifelse(ms<0, 0, ms) %>% as.numeric) %>% filter(ms>0)

write_csv(Super, "~/Documents/MSW/super_market_share.csv")
save(Super, file = "~/Documents/MSW/super_market_share.RData")
