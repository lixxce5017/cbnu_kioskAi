# Generated by Django 4.2.4 on 2023-08-15 15:01

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("bugger", "0003_alter_menu_type"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="menu",
            name="is_best_menu",
        ),
    ]