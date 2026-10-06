export PYTHONPATH=$HOME/Downloads/mininest/

#STEPSAMPLER=region-ortho-harm
STEPSAMPLER=cube-slice

SIN="multisine-0comp-2d multisine-1comp-5d multisine-2comp-8d multisine-3comp-11d"
HIGHD="asymgauss-100d corrfunnel-50d rosenbrock-50d"
#for problem in {corrfunnel-2d,asymgauss-4d,asymgauss-16d,corrfunnel-10d,rosenbrock-2d,rosenbrock-20d,eggbox-2d,beta-2d,beta-10d,beta-30d,loggamma-2d,loggamma-10d,loggamma-30d,box-5d}
for problem in {corrfunnel-2d,asymgauss-4d,rosenbrock-2d,eggbox-2d,beta-2d,loggamma-2d,box-5d,loggamma-10d,corrfunnel-10d,rosenbrock-20d,beta-10d} $SIN $HIGHD
do
	echo $problem
	#[ -e logs/${problem}-mlfriends-K400/plots/corner.pdf ] || 
	#{
	#	python3 runtoy.py --problem=$problem
	#}
	#ls logs/${problem}-${STEPSAMPLER}-K400-nsteps* 2>/dev/null >/dev/null && continue
	[ -e logs/${problem}-${STEPSAMPLER}-K400calibration.csv ] && continue
	python3 runtoy.py --problem=$problem --stepsampler=${STEPSAMPLER} | tee -a logs/${problem}-${STEPSAMPLER}-K400.log
done
