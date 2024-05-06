# Generated by Django 4.2.11 on 2024-05-06 13:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('MainPage', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='UploadedImage',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to='uploads/')),
                ('uploaded_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]