
STEPSAMPLER=cube-slice

SIN="multisine-0comp-2d multisine-1comp-5d multisine-2comp-8d multisine-3comp-11d"
HIGHD="asymgauss-100d corrfunnel-50d rosenbrock-50d"
for problem in {corrfunnel-2d,asymgauss-4d,rosenbrock-2d,eggbox-2d,beta-2d,loggamma-2d,box-5d,loggamma-10d,corrfunnel-10d,rosenbrock-20d,beta-10d} $SIN
do
	echo =================
	echo $problem
	echo
	d=$(echo ${problem}|sed 's,^.*-\([0-9]*\)d$,\1,g')
	python runqual.py $d logs/${problem}-${STEPSAMPLER}-K400-nsteps%d $d ${problem}-order.pdf

	for i in logs/${problem}-${STEPSAMPLER}-K400-nsteps*/plots/nsteps-calibration.pdf
	do
		ln -svf $i $(echo $i|sed 's,logs/\([^/-]*-[^/-]*\).*,\1,g')-Z.pdf
	done

	for i in logs/${problem}-${STEPSAMPLER}-K400-nsteps*/plots/nsteps-calibration-jumps.pdf
	do
		ln -svf $i $(echo $i|sed 's,logs/\([^/-]*-[^/-]*\).*,\1,g')-jumps.pdf
	done
	for i in logs/${problem}-${STEPSAMPLER}-K400-nsteps*/plots/ordertest.pdf
	do
		ln -svf $i $(echo $i|sed 's,logs/\([^/-]*-[^/-]*\).*,\1,g')-ordertest.pdf
	done
	for i in logs/${problem}-${STEPSAMPLER}-K400-nsteps{??,?}/plots/stepsampler.pdf.txt.gz.pdf
	do
		[ -e "$i" ] &&
		ln -svf $i $(echo $i|sed 's,logs/\([^/-]*-[^/-]*\).*,\1,g')-stepsampler-evol.pdf
	done
done
