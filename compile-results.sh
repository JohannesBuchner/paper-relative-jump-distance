LOGS=logs/
STEPSAMPLER=cube-slice

HIGHD="asymgaussH-100d corrfunnel-50d rosenbrock-50d"
for problem in {asymgaussH-4d,asymgaussH-16d,corrfunnel-2d,rosenbrock-2d,eggbox-2d,loggamma-2d,loggamma-10d,corrfunnel-10d,corrfunnel-20d,rosenbrock-20d}
do
	echo =================
	echo $problem
	echo
	d=$(echo ${problem}|sed 's,^.*-\([0-9]*\)d$,\1,g')
	echo $d
	python runqual.py $d ${LOGS}/${problem}-${STEPSAMPLER}-K400-seed42-nsteps%d $d ${problem}-order.pdf

	for i in ${LOGS}/${problem}-${STEPSAMPLER}-K400-seed42-nsteps*/plots/nsteps-calibration.pdf
	do
		ln -svf $i $(echo $i|sed "s,${LOGS}/"'\([^/-]*-[^/-]*\).*,\1,g')-Z.pdf
	done

	for i in ${LOGS}/${problem}-${STEPSAMPLER}-K400-seed42-nsteps*/plots/nsteps-calibration-jumps.pdf
	do
		ln -svf $i $(echo $i|sed "s,${LOGS}/"'\([^/-]*-[^/-]*\).*,\1,g')-jumps.pdf
	done
	for i in ${LOGS}/${problem}-${STEPSAMPLER}-K400-seed42-nsteps*/plots/ordertest.pdf
	do
		ln -svf $i $(echo $i|sed "s,${LOGS}/"'\([^/-]*-[^/-]*\).*,\1,g')-ordertest.pdf
	done
	for i in ${LOGS}/${problem}-${STEPSAMPLER}-K400-seed42-nsteps{??,?}/plots/stepsampler.pdf.txt.gz.pdf
	do
		[ -e "$i" ] &&
		ln -svf $i $(echo $i|sed "s,${LOGS}/"'\([^/-]*-[^/-]*\).*,\1,g')-stepsampler-evol.pdf
	done
	break
done
