# Under ImageNet folder
mkdir -p val
mkdir -p train

tar -xf ILSVRC2012_img_train.tar -C train
tar -xf ILSVRC2012_img_train_t3.tar -C train
tar -xf ILSVRC2012_img_val.tar -C val

cd train
for x in *.tar
for x in *.tar; do 
	fn="basename $x .tar" 
	mkdir $fn 
	tar -xf $x -C $fn 
	rm -f $fn.tar 
done

filename=valprep.sh
if [ -f "$filename" ];
then
    echo "$filename has found."
else
    echo "$filename has not been found"
    exit 1
fi

bash valprep.sh ./val