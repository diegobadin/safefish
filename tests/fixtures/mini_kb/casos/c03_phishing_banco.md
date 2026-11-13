---
id: FX-C03
modalidad: phishing_bancario
canal: [sms, email]
entidad_suplantada: banco
campaign_id: fx-kb-phishing-banco
fuente_url: https://example.org/fixtures/phishing-banco-ejemplo
---
Phishing bancario por SMS o mail que suplanta al Banco Ejemplo.

Descripción: un SMS o correo avisa que la cuenta fue bloqueada, que hay un consumo no reconocido o que hay que "validar los datos" para evitar la suspensión. Incluye un link acortado o un dominio parecido al oficial (por ejemplo, bancoejemplo-seguridad.com) que lleva a un formulario idéntico al home banking.

Frases típicas: "su cuenta será suspendida en 24 hs"; "ingrese al siguiente enlace para validar su identidad"; "detectamos un movimiento inusual, confirme sus datos".

Mecanismo: en el sitio falso se piden usuario, clave, datos de la tarjeta y a veces el token o código que llega por SMS. Con eso el atacante opera la cuenta o da de alta transferencias.
