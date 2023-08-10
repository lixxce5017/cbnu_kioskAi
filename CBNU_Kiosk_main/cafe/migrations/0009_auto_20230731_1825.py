# Generated by Django 3.2 on 2023-07-31 09:25

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('cafe', '0008_alter_menu_image'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='customer',
            name='image',
        ),
        migrations.RemoveField(
            model_name='customer',
            name='updated',
        ),
        migrations.RemoveField(
            model_name='customer',
            name='version',
        ),
        migrations.AddField(
            model_name='menu',
            name='en_title',
            field=models.CharField(default='', max_length=250),
        ),
        migrations.AlterField(
            model_name='customer',
            name='age_group',
            field=models.CharField(choices=[('10', '10대'), ('20', '20대'), ('30', '30대'), ('40', '40대'), ('50', '50대'), ('60', '60대 이상')], default=0, max_length=12),
        ),
        migrations.AlterField(
            model_name='menu',
            name='type',
            field=models.CharField(choices=[('hot_coffee', 'Hot Coffee'), ('ice_coffee', 'Ice Coffee'), ('non_coffee', 'Non Coffee'), ('smoothie', 'Smoothie'), ('bread', 'Bread'), ('cookie', 'Cookie')], default=0, max_length=20),
        ),
        migrations.AlterField(
            model_name='order',
            name='customer',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='cafe.customer'),
        ),
    ]
