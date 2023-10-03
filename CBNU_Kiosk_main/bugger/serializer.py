from rest_framework import serializers
from .models import Customer
from .models import Menu
class CustomerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Customer
        fields = ('__all__')


from .models import Menu

class MenuSerializer(serializers.ModelSerializer):
    class Meta:
        model = Menu
        fields = '__all__'
