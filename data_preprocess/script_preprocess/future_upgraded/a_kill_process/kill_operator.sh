

for i in {1..200}
do
    ps aux | grep operator_futures | awk '{print $2}' | xargs kill -9
    sleep 0.5 # 加入一个睡眠时间，防止过于频繁的执行导致系统负载过高
done
