# Generated by Django 5.0.4 on 2024-04-29 23:43

import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ehrapp', '0005_remove_individual_designation_alter_individual_dob_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='individual',
            name='DOB',
            field=models.DateField(default=django.utils.timezone.now),
        ),
    ]
