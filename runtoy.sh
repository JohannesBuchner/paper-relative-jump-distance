export PYTHONPATH=$HOME/Downloads/mininest/

STEPSAMPLER3=region-rwm
STEPSAMPLER2=de-mix
STEPSAMPLER=cube-slice
STEPSAMPLER4=region-slice

MIDD="corrfunnelN-20d rosenbrock-20d"
HIGHD="asymgaussH-100d"
seed=42
for problem in {asymgaussH-4d,asymgauss-16d,rosenbrock-2d,eggbox-2d,loggamma-2d,box-5d,loggamma-10d,corrfunnelN-10d,corrfunnelN-2d,corrfunnelN-5d,box-5d} $MIDD $HIGHD
do
	echo $problem
	#[ -e logs/${problem}-mlfriends-K400/plots/corner.pdf ] || 
	#{
	#	python3 runtoy.py --problem=$problem
	#}
	#ls logs/${problem}-${STEPSAMPLER}-K400-nsteps* 2>/dev/null >/dev/null && continue
	[ -e logs/${problem}-${STEPSAMPLER}-K400-seed${seed}calibration.csv ] && continue
	python3 runtoy.py --problem=$problem --stepsampler=${STEPSAMPLER} | tee -a logs/${problem}-${STEPSAMPLER}-K400-seed${seed}.log
done

exit 0
problem=corrfunnelN-10d
for K in 40 100 400 1000 4000
do
	[ -e logs/${problem}-${STEPSAMPLER}-K${K}-seed${seed}calibration.csv ] && continue
	python3 runtoy.py --num_live_points=${K} --problem=$problem --stepsampler=${STEPSAMPLER} | tee -a logs/${problem}-${STEPSAMPLER}-K${K}-seed${seed}.log
	[ -e logs/${problem}-${STEPSAMPLER2}-K${K}-seed${seed}calibration.csv ] && continue
	python3 runtoy.py --num_live_points=${K} --problem=$problem --stepsampler=${STEPSAMPLER2} | tee -a logs/${problem}-${STEPSAMPLER2}-K${K}-seed${seed}.log
done

for problem in eggbox-2d loggamma-2d
do
	for K in 1000
	do
		[ -e logs/${problem}-${STEPSAMPLER}-K${K}-seed${seed}calibration.csv ] && continue
		python3 runtoy.py --num_live_points=${K} --problem=$problem --stepsampler=${STEPSAMPLER} | tee -a logs/${problem}-${STEPSAMPLER}-K${K}-seed${seed}.log
	done
done

K=400
[ -e logs/${problem}-${STEPSAMPLER2}-K${K}-seed${seed}calibration.csv ] ||
python3 runtoy.py --num_live_points=${K} --problem=$problem --stepsampler=${STEPSAMPLER2} | tee -a logs/${problem}-${STEPSAMPLER2}-K${K}-seed${seed}.log
[ -e logs/${problem}-${STEPSAMPLER3}-K${K}-seed${seed}calibration.csv ] ||
python3 runtoy.py --num_live_points=${K} --problem=$problem --stepsampler=${STEPSAMPLER3} | tee -a logs/${problem}-${STEPSAMPLER3}-K${K}-seed${seed}.log
[ -e logs/${problem}-${STEPSAMPLER4}-K${K}-seed${seed}calibration.csv ] ||
python3 runtoy.py --num_live_points=${K} --problem=$problem --stepsampler=${STEPSAMPLER4} | tee -a logs/${problem}-${STEPSAMPLER4}-K${K}-seed${seed}.log

for seed in 1 2 3 4
do
	K=400
	[ -e logs/${problem}-${STEPSAMPLER}-K${K}-seed${seed}calibration.csv ] && continue
	python3 runtoy.py --num_live_points=${K} --seed=${seed} --problem=$problem --stepsampler=${STEPSAMPLER} | tee -a logs/${problem}-${STEPSAMPLER}-K${K}-seed${seed}.log
done
