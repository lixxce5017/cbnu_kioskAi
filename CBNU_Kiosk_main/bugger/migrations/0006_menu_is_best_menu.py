# Generated by Django 4.2.4 on 2023-10-09 15:32

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("bugger", "0005_alter_menu_type"),
    ]

    operations = [
        migrations.AddField(
            model_name="menu",
            name="is_best_menu",
            field=models.BooleanField(default=False),
        ),
    ]
