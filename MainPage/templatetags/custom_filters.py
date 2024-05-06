from django import template

register = template.Library()

@register.filter(name='format_price')
def format_price(value):
    """
    Ürün fiyatını istenen formatta biçimlendiren özel filtre.
    """
    if value == int(value):  # Eğer fiyat tam sayı ise
        return "{:.0f}".format(value)  # Virgülden sonrasını gösterme
    else:
        return "{:.2f}".format(value)  # Virgülden sonrasını iki basamak gösterme
