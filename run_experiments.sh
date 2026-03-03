for i in {1..3}
do


python simulation_migration.py -folder "uniform0run${i}"  -gradient 0 -highgradientstrength 0 -initialattractant 0 -plotting 0 -saving 500 &

done

for i in {1..3}
do


python simulation_migration.py -folder "uniform03run${i}"  -gradient 0 -highgradientstrength 0 -initialattractant 1.50E-10 -plotting 0 -saving 500 &

done

for i in {1..3}
do


python simulation_migration.py -folder "uniform3run${i}"  -gradient 0 -highgradientstrength 0 -initialattractant 1.50E-9 -plotting 0 -saving 500 &

done

for i in {1..3}
do


python simulation_migration.py -folder "uniform10run${i}"  -gradient 0 -highgradientstrength 0 -initialattractant 5.00E-09 -plotting 0 -saving 500 &

done

for i in {1..3}
do


python simulation_migration.py -folder "uniform30run${i}"  -gradient 0 -highgradientstrength 0 -initialattractant 1.50E-8 -plotting 0 -saving 500 &

done

for i in {1..3}
do
python simulation_migration.py -folder "uniform100run${i}"  -gradient 0 -highgradientstrength 0 -initialattractant 5.00E-8 -plotting 0 -saving 500 &

done

for i in {1..3}
do


python simulation_migration.py -folder "gradient10run${i}"  -gradient 1 -highgradientstrength 5.00E-09 -initialattractant 0 -plotting 0 -saving 500 &

done


wait




