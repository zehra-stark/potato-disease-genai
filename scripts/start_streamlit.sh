#!/bin/bash
cd /home/ec2-user/potato-app
sudo chown -R ec2-user:ec2-user /home/ec2-user/potato-app
chmod -R 755 /home/ec2-user/potato-app
pkill -f streamlit || true
nohup ~/.local/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0 > streamlit.log 2>&1 &
