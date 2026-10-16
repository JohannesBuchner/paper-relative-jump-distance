# everything everything (for latex table)
python3 replotruns.py $(for problem in logs/asymgaussH-4d-cube-slice-K400 logs/asymgaussH-16d-cube-slice-K400-seed42 logs/asymgaussH-100d-cube-slice-K400-seed42  logs/box-5d-cube-slice-K400 logs/rosenbrock-2d-cube-slice-K400 logs-new/rosenbrock-20d-cube-slice-K400 logs/eggbox-2d-cube-slice-K400-seed42 logs-new/loggamma-2d-cube-slice-K400-seed42 logs/loggamma-10d-cube-slice-K400-seed42 logs/corrfunnelN-10d-cube-slice-K{40,100,400,1000,4000}-seed42 logs/corrfunnelN-10d-cube-slice-K400-seed{1,2,3,4}-nsteps logs/corrfunnelN-10d-region-slice-K400-seed42-nsteps20 logs/corrfunnelN-10d-de-mix-K400-seed42-nsteps20 logs/corrfunnelN-10d-region-rwm-K400-seed42-nsteps20 logs-prev/eightschools-10d-cube-slice-K400 systematiclogs-prev/ligo/ultranest-calibrator/ ; do echo ${problem}*/plots/stepsampler.pdf.txt.gz; done)

# 
python3 replotruns.py $(for problem in logs/asymgaussH-4d-cube-slice-K400 logs/asymgaussH-16d-cube-slice-K400-seed42 logs/asymgaussH-100d-cube-slice-K400-seed42  logs/box-5d-cube-slice-K400 logs/rosenbrock-2d-cube-slice-K400 logs-new/rosenbrock-20d-cube-slice-K400 logs/eggbox-2d-cube-slice-K400-seed42 logs-new/loggamma-2d-cube-slice-K400-seed42 logs/loggamma-10d-cube-slice-K400-seed42 logs/corrfunnelN-10d-cube-slice-K400-seed42 logs-prev/eightschools-10d-cube-slice-K400 systematiclogs-prev/ligo/ultranest-calibrator/; do echo ${problem}*/plots/stepsampler.pdf.txt.gz; done)


#python3 replotruns_variants.py logs/asymgauss-{4,16,100}d-cube-slice-K400-seed42-nsteps*/plots/stepsampler.pdf.txt.gz
python3 replotruns_variants.py logs/asymgaussH-{4,16,100}d-cube-slice-K400-seed42-nsteps*/plots/stepsampler.pdf.txt.gz

# python3 replotruns_variants.py logs/corrfunnelN-{10,50}d-cube-slice-K400-seed42-nsteps*/plots/stepsampler.pdf.txt.gz


python3 replotruns_variants.py logs/corrfunnelN-10d-cube-slice-K400-seed{1,2,3,4}-nsteps*/plots/stepsampler.pdf.txt.gz

# python3 replotruns_variants.py logs/corrfunnelN-10d-cube-slice-K{40,100,400,1000,4000}-*seed42-nsteps*/plots/stepsampler.pdf.txt.gz
python replotruns_varyK.py logs/corrfunnelN-10d-cube-slice-K{40,100,400,1000,4000}-*seed42-nsteps10/plots/stepsampler.pdf.txt.gz

#python3 replotruns_varysampler.py logs/corrfunnelN-10d-*-K400-seed42-nsteps20/plots/stepsampler.pdf.txt.gz
python3 replotruns_varysampler.py logs/corrfunnelN-10d-{cube-slice,de-mix,region-slice}-K400-seed42-nsteps20/plots/stepsampler.pdf.txt.gz logs/corrfunnelN-10d-region-rwm-K400-seed42-nsteps160/plots/stepsampler.pdf.txt.gz

# python3 replotruns_varysampler.py logs/rosenbrock-20d-*-K400-nsteps20/plots/stepsampler.pdf.txt.gz
