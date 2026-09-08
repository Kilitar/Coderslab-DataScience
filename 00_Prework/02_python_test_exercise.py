"""
Python Test Exercise - Prework
Machine Learning Course - CodersLab

Úkoly:
1. Funkce pro základní matematické operace (add, sub, mul, div) s uspořádáním parametrů a zaokrouhlením.
2. Funkce pro filtrování jmen začínajících na písmeno 'C'.
3. Funkce pro filtrování zaměstnanců podle platu nad stanovenou hranici.
"""

from typing import Dict, List, Tuple, Union


# ==============================================================================
# ÚKOL 1: Matematické operace s podmínkami a zaokrouhlením
# ==============================================================================
def calculate(a: Union[int, float], b: Union[int, float], math_operation: str) -> float:
    """
    Provede matematickou operaci se dvěma reálnými čísly.
    U operací závislých na pořadí (sub, div) je na prvním místě větší z čísel.
    Výsledek zaokrouhluje na 2 desetinná místa.
    """
    larger = max(a, b)
    smaller = min(a, b)

    if math_operation == "add":
        result = larger + smaller
    elif math_operation == "sub":
        result = larger - smaller
    elif math_operation == "mul":
        result = larger * smaller
    elif math_operation == "div":
        if smaller == 0:
            raise ZeroDivisionError("Dělení nulou není povoleno.")
        result = larger / smaller
    else:
        raise ValueError(
            f"Neznámá operace: '{math_operation}'. Povolené operace jsou: add, sub, mul, div"
        )

    return round(result, 2)


# ==============================================================================
# ÚKOL 2: Filtrování jmen začínajících písmenem 'C'
# ==============================================================================
def filter_c_names(names: List[str]) -> List[str]:
    """
    Vrátí pouze ta jména ze seznamu, která začínají na písmeno 'C'.
    """
    return [name for name in names if name.startswith("C")]


# ==============================================================================
# ÚKOL 3: Filtrování zaměstnanců podle výše mzdy
# ==============================================================================
def filter_employees_by_salary(
    employees: Dict[str, Tuple[str, Union[int, float]]],
    threshold: Union[int, float]
) -> Dict[str, Tuple[str, Union[int, float]]]:
    """
    Vrátí slovník zaměstnanců, jejichž plat je vyšší než zadaná částka (threshold).
    employees: slovník {jméno: (pozice, plat)}
    threshold: minimální plat (vrací pouze ty s platem > threshold)
    """
    return {
        name: data
        for name, data in employees.items()
        if data[1] > threshold
    }


# ==============================================================================
# Otestování funkcí
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TEST ÚKOLU 1: calculate()")
    print("=" * 60)
    print(f"calculate(2, 5, 'add'): {calculate(2, 5, 'add')} (očekáváno 7.0)")
    print(f"calculate(3, 10, 'sub'): {calculate(3, 10, 'sub')} (očekáváno 10 - 3 = 7.0)")
    print(f"calculate(4, 2.5, 'mul'): {calculate(4, 2.5, 'mul')} (očekáváno 10.0)")
    print(f"calculate(3, 10, 'div'): {calculate(3, 10, 'div')} (očekáváno 10 / 3 = 3.33)")

    print("\n" + "=" * 60)
    print("TEST ÚKOLU 2: filter_c_names()")
    print("=" * 60)
    royal_family_names = [
        "Elizabeth", "Phillip", "Charles", "Camilla",
        "William", "Catherine", "Anne", "Harry", "Meghan"
    ]
    c_names = filter_c_names(royal_family_names)
    print(f"Vstupní seznam: {royal_family_names}")
    print(f"Výsledek: {c_names}")

    print("\n" + "=" * 60)
    print("TEST ÚKOLU 3: filter_employees_by_salary()")
    print("=" * 60)
    company_employees = {
        "Jan Novák": ("Data Analyst", 55000),
        "Petr Svoboda": ("Junior Python Developer", 45000),
        "Eva Černá": ("Senior ML Engineer", 120000),
        "Lucie Bílá": ("Product Owner", 85000),
        "Martin Dvořák": ("QA Specialist", 40000),
        "Karel Procházka": ("Data Scientist", 95000)
    }
    salary_threshold = 60000
    filtered_employees = filter_employees_by_salary(company_employees, salary_threshold)
    print(f"Zaměstnanci s platem vyšším než {salary_threshold} Kč:")
    for name, (position, salary) in filtered_employees.items():
        print(f" - {name}: {position} ({salary:,} Kč)")
