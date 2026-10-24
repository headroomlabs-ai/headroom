// Datas sempre como texto ISO (AAAA-MM-DD), sem fuso horário.
export const hojeISO = () => {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
};

export const ehISO = (s) => typeof s === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(s) && !Number.isNaN(Date.parse(s));

export function addMeses(iso, n) {
  const [a, m, d] = iso.split('-').map(Number);
  const total = a * 12 + (m - 1) + n;
  const ano = Math.floor(total / 12);
  const mes = (total % 12) + 1;
  const ultimo = new Date(ano, mes, 0).getDate();
  return `${ano}-${String(mes).padStart(2, '0')}-${String(Math.min(d, ultimo)).padStart(2, '0')}`;
}

export const mesDe = (iso) => iso.slice(0, 7);

// Lista de meses (AAAA-MM) entre duas datas, inclusive.
export function mesesEntre(de, ate) {
  const out = [];
  let [a, m] = de.split('-').map(Number);
  const [fa, fm] = ate.split('-').map(Number);
  while (a < fa || (a === fa && m <= fm)) {
    out.push(`${a}-${String(m).padStart(2, '0')}`);
    m++;
    if (m > 12) { m = 1; a++; }
  }
  return out;
}
