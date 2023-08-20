# Generated by Django 4.2.4 on 2023-08-15 14:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("bugger", "0002_alter_menu_type"),
    ]

    operations = [
        migrations.AlterField(
            model_name="menu",
            name="type",
            field=models.CharField(
                choices=[
                    ("bugger", " bugger"),
                    ("Premium", "Premium"),
                    ("drink", "drink"),
                    ("side", "side"),
                    ("Whopper", "Whopper"),
                ],
                default=0,
                max_length=20,
            ),
        ),
    ]