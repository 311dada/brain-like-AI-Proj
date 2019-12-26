for x in exp/*/train.log; do 
echo $x
cat $x |grep Eval 
echo ""
done
