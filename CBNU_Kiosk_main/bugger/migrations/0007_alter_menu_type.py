# Generated by Django 4.2.4 on 2023-10-19 02:21

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("bugger", "0006_menu_is_best_menu"),
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
                    ("set_change", "set_change"),
                    ("old_bugger", "old_bugger"),
                    ("old_Premium", "old_Premium"),
                    ("old_Whopper", "old_Whopper"),
                ],
                default=0,
                max_length=20,
            ),
        ),
    ]
