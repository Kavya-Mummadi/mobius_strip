from datetime import datetime 
now = datetime.now()
f = now.strftime("%d/%m/%Y","%H:%M:%S")
print(type(f))