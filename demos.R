library(dplyr)
library(rstanarm)
library(AER)



# A basic
data("CigarettesSW")


CigData <- CigarettesSW %>%
  mutate(rprice = price/cpi, # Real price adjusted for inflation
         rincome = income/population/cpi, # Real per-capital income
         tdiff = (taxs - tax)/cpi,
         year = as.numeric(as.character(year)),
         year = year - min(year))

mod1 <- stan_lm(log(packs) ~ log(rprice), 
                data = CigData, 
                prior = R2(0.3, what = "median"), 
                cores = 4)

mod2 <- stan_lm(log(packs) ~ log(rprice) + log(income) + year, 
                data = CigData, 
                prior = R2(0.5, what = "median"), 
                cores = 4,
                iter = 500)


CigPanel <- CigData %>%
  group_by(state) %>%
  arrange(state, year) %>%
  summarise(d_packs = diff(log(packs)),
            d_price = diff(log(rprice)),
            d_pop = diff(log(population)),
            d_income = diff(log(income)),
            d_taxes = diff(taxs))

mod3 <- mod2 <- stan_lm(d_packs ~ d_price + d_packs + d_pop, 
                        data = CigPanel, 
                        prior = R2(0.3, what = "mean"), 
                        cores = 4,
                        iter = 500)

