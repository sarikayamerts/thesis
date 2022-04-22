require(arrow)
require(data.table)
require(lubridate)
require(mgcv)
require(gratia)

data_path='C:/Mustafa/Research/Wind/data/2019-01-24_outlier_removed.parquet'


dat=data.table(read_parquet(data_path))
dat[,date_time:=as_datetime(epoch,tz='Turkey')]
dat[,date:=date(date_time)]
dat[,hour:=hour(date_time)]

dat[,ws_SW:=sqrt(UGRD_80.m.above.ground.SW^2 + VGRD_80.m.above.ground.SW^2)]
dat[,wdir_SW:=atan2(UGRD_80.m.above.ground.SW/ws_SW, VGRD_80.m.above.ground.SW/ws_SW)* 180/pi ]
dat[,ws_NW:=sqrt(UGRD_80.m.above.ground.NW^2 + VGRD_80.m.above.ground.NW^2)]
dat[,wdir_NW:=atan2(UGRD_80.m.above.ground.NW/ws_NW, VGRD_80.m.above.ground.NW/ws_NW)* 180/pi ]
dat[,ws_NE:=sqrt(UGRD_80.m.above.ground.NE^2 + VGRD_80.m.above.ground.NE^2)]
dat[,wdir_NE:=atan2(UGRD_80.m.above.ground.NE/ws_NE, VGRD_80.m.above.ground.NE/ws_NE)* 180/pi ]
dat[,ws_SE:=sqrt(UGRD_80.m.above.ground.SE^2 + VGRD_80.m.above.ground.SE^2)]
dat[,wdir_SE:=atan2(UGRD_80.m.above.ground.SE/ws_SE, VGRD_80.m.above.ground.SE/ws_SE)* 180/pi ]

dat[,max_prod:=max(production_cleaned),by=list(rt_plant_id)]
dat[,utilization:=production_cleaned/max_prod]

all_plants=dat[,list(avg_prod=mean(production_cleaned),max_prod=max(production_cleaned)),by=list(rt_plant_id )]
all_plants=all_plants[order(-avg_prod)]

train_ratio=0.7

i=1
#for(i in 1:nrow(all_plants)){
	selected_plant=all_plants$rt_plant_id[i]
	filtered_data=dat[rt_plant_id==selected_plant]
	
	train_ids=c(1:floor(0.7*nrow(filtered_data)))
	train_data=filtered_data[train_ids]
	test_data=filtered_data[-train_ids]
	
	# a base linear regression model (commonly used) ignores direction info
	
	lr_fit=glm(production_cleaned~poly(ws_SW,3)+poly(ws_NW,3)+poly(ws_NE,3)+poly(ws_SE,3),train_data,family=gaussian)
	# just checking
	summary(lr_fit)
	predicted=predict(lr_fit,test_data)
	predicted[predicted<0]=0
	
	results=copy(test_data[,list(date,hour,production_cleaned,max_prod,utilization)])
	results[,pred:=predicted]
	results[,model:='poly_reg_gaussian']
	
	lr_fit=glm(utilization~poly(ws_SW,3)+poly(ws_NW,3)+poly(ws_NE,3)+poly(ws_SE,3),train_data,family=binomial)
	# just checking
	summary(lr_fit)
	predicted=predict(lr_fit,test_data,type='response')
	
	tmp=copy(test_data[,list(date,hour,production_cleaned,max_prod,utilization)])
	tmp[,pred:=predicted*max_prod]
	tmp[,model:='poly_reg_binomial']
	
	results=rbindlist(list(results,tmp))
	
	# GAM models
	gam_fit=bam(utilization~te(UGRD_80.m.above.ground.SW,VGRD_80.m.above.ground.SW,bs=c('tp','tp'))+
							te(UGRD_80.m.above.ground.NW,VGRD_80.m.above.ground.NW,bs=c('tp','tp'))+
							te(UGRD_80.m.above.ground.NE,VGRD_80.m.above.ground.NE,bs=c('tp','tp'))+
							te(UGRD_80.m.above.ground.SE,VGRD_80.m.above.ground.SE,bs=c('tp','tp'))
							,train_data,family=binomial,discrete=T,method='fREML')
							   
	draw(gam_fit)
	predicted=predict(gam_fit,test_data,type='response')
	
	tmp=copy(test_data[,list(date,hour,production_cleaned,max_prod,utilization)])
	tmp[,pred:=predicted*max_prod]
	tmp[,model:='gam_uv_binomial']
	
	results=rbindlist(list(results,tmp))
	
	gam_fit=bam(utilization~te(ws_SW,wdir_SW,bs=c('tp','cc'))+
							te(ws_NW,wdir_NW,bs=c('tp','cc'))+
							te(ws_NE,wdir_NE,bs=c('tp','cc'))+
							te(ws_SE,wdir_SE,bs=c('tp','cc')),train_data,family=binomial,
							   discrete=T,method='fREML')
							   
	draw(gam_fit)
	predicted=predict(gam_fit,test_data,type='response')
	
	tmp=copy(test_data[,list(date,hour,production_cleaned,max_prod,utilization)])
	tmp[,pred:=predicted*max_prod]
	tmp[,model:='gam_ws_binomial']
	
	results=rbindlist(list(results,tmp))
	
	gam_fit=bam(utilization~te(ws_SW,wdir_SW,bs=c('tp','cc'))+
							te(ws_NW,wdir_NW,bs=c('tp','cc'))+
							te(ws_NE,wdir_NE,bs=c('tp','cc'))+
							te(ws_SE,wdir_SE,bs=c('tp','cc')),train_data,family=quasibinomial,
							   discrete=T,method='fREML')
							   
	draw(gam_fit)
	predicted=predict(gam_fit,test_data,type='response')
	
	tmp=copy(test_data[,list(date,hour,production_cleaned,max_prod,utilization)])
	tmp[,pred:=predicted*max_prod]
	tmp[,model:='gam_ws_quasibinomial']
	
	results=rbindlist(list(results,tmp))
	
	gam_fit=bam(production_cleaned~te(ws_SW,wdir_SW,bs=c('tp','cc'))+
							te(ws_NW,wdir_NW,bs=c('tp','cc'))+
							te(ws_NE,wdir_NE,bs=c('tp','cc'))+
							te(ws_SE,wdir_SE,bs=c('tp','cc')),train_data,family=gaussian,
							   discrete=T,method='fREML')
							   
	draw(gam_fit)
	predicted=predict(gam_fit,test_data,type='response')
	
	tmp=copy(test_data[,list(date,hour,production_cleaned,max_prod,utilization)])
	predicted[predicted<0]=0
	tmp[,pred:=predicted]
	tmp[,model:='gam_ws_qaussian']
	
	results=rbindlist(list(results,tmp))
	
	
	performance=results[,list(wmape=sum(abs(pred-production_cleaned))/sum(production_cleaned),
							  bias=sum(pred-production_cleaned)/sum(production_cleaned),
							  avg_prod=mean(production_cleaned),
							  n_obs=.N), by=list(model)]
	print(performance)
	

	
	
#}