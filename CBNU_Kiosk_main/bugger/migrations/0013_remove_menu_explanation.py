# Generated by Django 4.2.4 on 2023-10-31 15:13

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("bugger", "0012_menu_explanation"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="menu",
            name="explanation",
        ),
    ]
