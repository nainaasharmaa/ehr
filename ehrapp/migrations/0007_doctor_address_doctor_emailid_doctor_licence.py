# Generated by Django 5.0.4 on 2024-04-30 00:11

import django.core.validators
import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ehrapp', '0006_alter_individual_dob'),
    ]

    operations = [
        migrations.AddField(
            model_name='doctor',
            name='address',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='doctor',
            name='emailid',
            field=models.EmailField(default=django.utils.timezone.now, max_length=254),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='doctor',
            name='licence',
            field=models.ImageField(default=django.utils.timezone.now, upload_to='ehr/static', validators=[django.core.validators.FileExtensionValidator(allowed_extensions=['jpg', 'jpeg', 'png'])]),
            preserve_default=False,
        ),
    ]
