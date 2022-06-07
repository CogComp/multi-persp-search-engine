from django import template

register = template.Library()


@register.filter
def split_underscore_join(str):
    return "_".join(str.split())


@register.filter
def lower(str):
    return str.lower()