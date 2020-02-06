
DATA_DIRS=../data/eval_result/*iter_200
for DIR in $DATA_DIRS
do
    echo "$DIR"
    # count numbers
    echo "counts: "
    ls $DIR| wc -l
    ls $DIR| grep _1 | wc -l 
    ls $DIR| grep -v _1 | wc -l 

    # model corrent number
    echo "model:"
    grep "model" $DIR/* | wc -l 
    grep "model" $DIR/* | grep _1 | wc -l
    grep "model" $DIR/* | grep -v _1 | wc -l

    # depth 1 correct number
    echo "exploit total:"
    grep "exploit" $DIR/* | wc -l 
    echo "exploit 2: "
    grep "exploit: 2" $DIR/* | wc -l 
    echo "exploit 1: "
    grep "exploit" $DIR/* | grep -v "exploit: 2" | wc -l 

    echo "fails:"
    grep "\[\]" $DIR/* | wc -l 
    grep "\[\]" $DIR/* | grep _1: | wc -l 
    grep "\[\]" $DIR/* | grep -v _1: | wc -l

    echo "process:"
    ls $DIR| grep _1 | wc -l 
    echo "out of "
    grep "\[\]" $DIR/* | grep -v _1: | wc -l
done