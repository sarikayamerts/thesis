rm(list=ls());gc(); #Clean workspace
options(stringsAsFactors=FALSE)
options(dplyr.width=Inf)
options(scipen=999)
Sys.setenv(TZ='UTC')
options(digits.secs=3)
options(pillar.subtle=FALSE)
library(data.table)
library(lubridate)
library(tidyverse)
library(EPIAS)
library(energysupport)
### cred
# put S3 credentials here
### cred

all_powerplants=get_generation_powerplant_list()$the_df

params = list()
org_bucket_name="producers-el"
params$energy_type_rt="wind"
params$train_start_date="2017-01-01"

meta = read.csv("/home/burak/Desktop/Research/Mert/res_meta.csv")

all_prod_df = tibble()

for(i in 1:nrow(meta)){
  print(i)
  try({
    prod_df <- EPIAS::get_plant_rt_production(start_date=params$train_start_date,
                                              end_date = lubridate::today("Turkey"),
                                              plant_id=  meta$rt_plant_id[i],
                                              energy_type=params$energy_type_rt)
    
    all_prod_df = rbind(all_prod_df %>% bind_rows(
      prod_df %>% mutate(rt_plant_id =  meta$rt_plant_id[i] )
    ))
  })
}

write.csv(all_prod_df,"/home/burak/Desktop/Research/Mert/production.csv")
