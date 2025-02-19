FROM python:3.10
COPY . /app 
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE $PORT
CMD unicorn --workers=4 --bind 0.0.0.0:$PORT app:app

# COPY . /app #copy all files from the current directory to /app
# WORKDIR /app #set the working directory to /app
# RUN pip install -r requirements.txt   #install the dependencies
# EXPOSE $PORT   #expose the port
# CMD unicorn --workers=4  #run the application inside the container(heroku uses unicorn as the web server), devide the requests between 4 workers
# --bind 0.0.0.0:$PORT app:app #bind the application to the port and run the app.py file
# port number will be binded to the local host 0.0.0.0 and the port number will be the one that is provided by the environment variable PORT