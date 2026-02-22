-- ==========================================
-- ESTRUCTURA + SEMILLA (MariaDB/MySQL)
-- Compatible con SOM3D-BACKEND/app/models.py
-- ==========================================

SET FOREIGN_KEY_CHECKS = 1;

-- -------------------------
-- Usuario
-- -------------------------
CREATE TABLE `Usuario` (
  `id_usuario` int NOT NULL AUTO_INCREMENT,
  `nombre` varchar(100) NOT NULL,
  `apellido` varchar(100) NOT NULL,
  `correo` varchar(100) NOT NULL,
  `contrasena` varchar(255) NOT NULL,
  `telefono` varchar(20) DEFAULT NULL,
  `direccion` varchar(255) DEFAULT NULL,
  `ciudad` varchar(100) DEFAULT NULL,
  `rol` enum('ADMINISTRADOR','MEDICO') NOT NULL,
  `activo` tinyint(1) NOT NULL DEFAULT 1,
  `creado_en` timestamp NOT NULL DEFAULT current_timestamp(),
  `actualizado_en` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  PRIMARY KEY (`id_usuario`),
  UNIQUE KEY `uq_usuario_correo` (`correo`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- -------------------------
-- Hospital
-- -------------------------
CREATE TABLE `Hospital` (
  `id_hospital` int NOT NULL AUTO_INCREMENT,
  `nombre` varchar(150) NOT NULL,
  `direccion` varchar(255) DEFAULT NULL,
  `ciudad` varchar(100) DEFAULT NULL,
  `telefono` varchar(30) DEFAULT NULL,
  `correo` varchar(100) DEFAULT NULL,
  `codigo` varchar(12) NOT NULL,
  `estado` enum('ACTIVO','INACTIVO') NOT NULL DEFAULT 'ACTIVO',
  `creado_en` timestamp NOT NULL DEFAULT current_timestamp(),
  `actualizado_en` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  PRIMARY KEY (`id_hospital`),
  UNIQUE KEY `uq_hospital_codigo` (`codigo`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- -------------------------
-- Medico
-- -------------------------
CREATE TABLE `Medico` (
  `id_medico` int NOT NULL AUTO_INCREMENT,
  `id_usuario` int NOT NULL,
  `id_hospital` int DEFAULT NULL,
  `referenciado` tinyint(1) NOT NULL DEFAULT 0,
  `estado` enum('ACTIVO','INACTIVO') NOT NULL DEFAULT 'ACTIVO',
  `creado_en` timestamp NOT NULL DEFAULT current_timestamp(),
  `actualizado_en` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  PRIMARY KEY (`id_medico`),
  UNIQUE KEY `uq_medico_id_usuario` (`id_usuario`),
  KEY `idx_medico_id_hospital` (`id_hospital`),
  CONSTRAINT `fk_medico_usuario` FOREIGN KEY (`id_usuario`) REFERENCES `Usuario` (`id_usuario`),
  CONSTRAINT `fk_medico_hospital` FOREIGN KEY (`id_hospital`) REFERENCES `Hospital` (`id_hospital`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- -------------------------
-- Plan
-- -------------------------
CREATE TABLE `Plan` (
  `id_plan` int NOT NULL AUTO_INCREMENT,
  `nombre` varchar(80) NOT NULL,
  `precio` decimal(12,2) NOT NULL,
  `periodo` enum('MENSUAL','TRIMESTRAL','ANUAL') NOT NULL,
  `duracion_meses` int NOT NULL,
  `creado_en` timestamp NOT NULL DEFAULT current_timestamp(),
  `actualizado_en` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  PRIMARY KEY (`id_plan`),
  UNIQUE KEY `uq_plan_nombre` (`nombre`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- -------------------------
-- Suscripcion
-- -------------------------
CREATE TABLE `Suscripcion` (
  `id_suscripcion` int NOT NULL AUTO_INCREMENT,
  `id_medico` int DEFAULT NULL,
  `id_hospital` int DEFAULT NULL,
  `id_plan` int NOT NULL,
  `fecha_inicio` datetime NOT NULL DEFAULT current_timestamp(),
  `fecha_expiracion` datetime DEFAULT NULL,
  `estado` enum('ACTIVA','PAUSADA') NOT NULL DEFAULT 'ACTIVA',
  `creado_en` timestamp NOT NULL DEFAULT current_timestamp(),
  `actualizado_en` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  PRIMARY KEY (`id_suscripcion`),
  KEY `idx_suscripcion_id_medico` (`id_medico`),
  KEY `idx_suscripcion_id_hospital` (`id_hospital`),
  KEY `idx_suscripcion_id_plan` (`id_plan`),
  KEY `idx_suscripcion_estado_exp` (`estado`,`fecha_expiracion`),
  CONSTRAINT `fk_suscripcion_medico` FOREIGN KEY (`id_medico`) REFERENCES `Medico` (`id_medico`),
  CONSTRAINT `fk_suscripcion_hospital` FOREIGN KEY (`id_hospital`) REFERENCES `Hospital` (`id_hospital`),
  CONSTRAINT `fk_suscripcion_plan` FOREIGN KEY (`id_plan`) REFERENCES `Plan` (`id_plan`),
  CONSTRAINT `chk_suscripcion_un_pagador` CHECK (((`id_medico` IS NOT NULL) <> (`id_hospital` IS NOT NULL)))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- -------------------------
-- Pago
-- -------------------------
CREATE TABLE `Pago` (
  `id_pago` int NOT NULL AUTO_INCREMENT,
  `id_suscripcion` int NOT NULL,
  `referencia_epayco` varchar(100) NOT NULL,
  `monto` decimal(12,2) NOT NULL,
  `fecha_pago` datetime NOT NULL DEFAULT current_timestamp(),
  `creado_en` timestamp NOT NULL DEFAULT current_timestamp(),
  `actualizado_en` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  PRIMARY KEY (`id_pago`),
  UNIQUE KEY `uq_pago_referencia_epayco` (`referencia_epayco`),
  KEY `idx_pago_id_suscripcion` (`id_suscripcion`),
  CONSTRAINT `fk_pago_suscripcion` FOREIGN KEY (`id_suscripcion`) REFERENCES `Suscripcion` (`id_suscripcion`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- -------------------------
-- Paciente
-- -------------------------
CREATE TABLE `Paciente` (
  `id_paciente` int NOT NULL AUTO_INCREMENT,
  `id_medico` int NOT NULL,
  `doc_tipo` varchar(20) DEFAULT NULL,
  `doc_numero` varchar(40) DEFAULT NULL,
  `nombres` varchar(100) NOT NULL,
  `apellidos` varchar(100) NOT NULL,
  `fecha_nacimiento` date DEFAULT NULL,
  `sexo` varchar(20) DEFAULT NULL,
  `telefono` varchar(30) DEFAULT NULL,
  `correo` varchar(120) DEFAULT NULL,
  `direccion` varchar(200) DEFAULT NULL,
  `ciudad` varchar(80) DEFAULT NULL,
  `estado` enum('ACTIVO','INACTIVO') NOT NULL DEFAULT 'ACTIVO',
  `creado_en` timestamp NOT NULL DEFAULT current_timestamp(),
  `actualizado_en` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  PRIMARY KEY (`id_paciente`),
  KEY `idx_paciente_id_medico` (`id_medico`),
  CONSTRAINT `fk_paciente_medico` FOREIGN KEY (`id_medico`) REFERENCES `Medico` (`id_medico`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- -------------------------
-- Estudio
-- -------------------------
CREATE TABLE `Estudio` (
  `id_estudio` int NOT NULL AUTO_INCREMENT,
  `id_paciente` int NOT NULL,
  `id_medico` int NOT NULL,
  `modalidad` varchar(20) DEFAULT NULL,
  `fecha_estudio` datetime NOT NULL DEFAULT current_timestamp(),
  `descripcion` varchar(200) DEFAULT NULL,
  `notas` text DEFAULT NULL,
  `creado_en` timestamp NOT NULL DEFAULT current_timestamp(),
  `actualizado_en` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  PRIMARY KEY (`id_estudio`),
  KEY `idx_estudio_id_paciente` (`id_paciente`),
  KEY `idx_estudio_id_medico` (`id_medico`),
  CONSTRAINT `fk_estudio_paciente` FOREIGN KEY (`id_paciente`) REFERENCES `Paciente` (`id_paciente`),
  CONSTRAINT `fk_estudio_medico` FOREIGN KEY (`id_medico`) REFERENCES `Medico` (`id_medico`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- -------------------------
-- JobConv
-- -------------------------
CREATE TABLE `JobConv` (
  `job_id` varchar(64) NOT NULL,
  `id_usuario` int NOT NULL,
  `status` enum('QUEUED','RUNNING','DONE','ERROR','CANCELED') NOT NULL DEFAULT 'QUEUED',
  `enable_ortopedia` tinyint(1) NOT NULL DEFAULT 1,
  `enable_appendicular` tinyint(1) NOT NULL DEFAULT 0,
  `enable_muscles` tinyint(1) NOT NULL DEFAULT 0,
  `enable_skull` tinyint(1) NOT NULL DEFAULT 0,
  `enable_teeth` tinyint(1) NOT NULL DEFAULT 0,
  `enable_hip_implant` tinyint(1) NOT NULL DEFAULT 0,
  `extra_tasks_json` text DEFAULT NULL,
  `queue_name` varchar(80) DEFAULT NULL,
  `started_at` datetime DEFAULT NULL,
  `finished_at` datetime DEFAULT NULL,
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  PRIMARY KEY (`job_id`),
  KEY `idx_jobconv_id_usuario` (`id_usuario`),
  CONSTRAINT `fk_jobconv_usuario` FOREIGN KEY (`id_usuario`) REFERENCES `Usuario` (`id_usuario`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- -------------------------
-- JobSTL
-- -------------------------
CREATE TABLE `JobSTL` (
  `id_jobstl` int NOT NULL AUTO_INCREMENT,
  `job_id` varchar(64) NOT NULL,
  `id_paciente` int DEFAULT NULL,
  `stl_size` bigint DEFAULT NULL,
  `num_stl_archivos` int DEFAULT NULL,
  `notas` text DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  PRIMARY KEY (`id_jobstl`),
  KEY `idx_jobstl_job_id` (`job_id`),
  KEY `idx_jobstl_id_paciente` (`id_paciente`),
  CONSTRAINT `fk_jobstl_jobconv` FOREIGN KEY (`job_id`) REFERENCES `JobConv` (`job_id`),
  CONSTRAINT `fk_jobstl_paciente` FOREIGN KEY (`id_paciente`) REFERENCES `Paciente` (`id_paciente`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- -------------------------
-- VisorEstado
-- -------------------------
CREATE TABLE `VisorEstado` (
  `id_visor_estado` int NOT NULL AUTO_INCREMENT,
  `id_medico` int NOT NULL,
  `id_paciente` int DEFAULT NULL,
  `id_jobstl` int DEFAULT NULL,
  `titulo` varchar(200) NOT NULL,
  `descripcion` varchar(400) DEFAULT NULL,
  `ui_json` text NOT NULL,
  `modelos_json` text NOT NULL,
  `notas_json` text NOT NULL,
  `i18n_json` text NOT NULL,
  `creado_en` timestamp NOT NULL DEFAULT current_timestamp(),
  `actualizado_en` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  PRIMARY KEY (`id_visor_estado`),
  KEY `idx_visor_id_medico` (`id_medico`),
  KEY `idx_visor_id_paciente` (`id_paciente`),
  KEY `idx_visor_id_jobstl` (`id_jobstl`),
  CONSTRAINT `fk_visor_medico` FOREIGN KEY (`id_medico`) REFERENCES `Medico` (`id_medico`),
  CONSTRAINT `fk_visor_paciente` FOREIGN KEY (`id_paciente`) REFERENCES `Paciente` (`id_paciente`),
  CONSTRAINT `fk_visor_jobstl` FOREIGN KEY (`id_jobstl`) REFERENCES `JobSTL` (`id_jobstl`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- -------------------------
-- mensaje
-- -------------------------
CREATE TABLE `mensaje` (
  `id_mensaje` int NOT NULL AUTO_INCREMENT,
  `id_medico` int NOT NULL,
  `id_paciente` int DEFAULT NULL,
  `tipo` varchar(30) NOT NULL,
  `titulo` varchar(200) NOT NULL,
  `descripcion` text NOT NULL,
  `severidad` varchar(20) NOT NULL,
  `adjunto_url` varchar(500) DEFAULT NULL,
  `estado` varchar(30) NOT NULL DEFAULT 'nuevo',
  `respuesta_admin` text DEFAULT NULL,
  `leido_admin` tinyint(1) NOT NULL DEFAULT 0,
  `leido_medico` tinyint(1) NOT NULL DEFAULT 0,
  `creado_en` datetime NOT NULL DEFAULT current_timestamp(),
  `actualizado_en` datetime NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  PRIMARY KEY (`id_mensaje`),
  KEY `idx_mensaje_id_medico` (`id_medico`),
  KEY `idx_mensaje_id_paciente` (`id_paciente`),
  CONSTRAINT `fk_mensaje_medico` FOREIGN KEY (`id_medico`) REFERENCES `Medico` (`id_medico`),
  CONSTRAINT `fk_mensaje_paciente` FOREIGN KEY (`id_paciente`) REFERENCES `Paciente` (`id_paciente`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- -------------------------
-- HospitalCode
-- -------------------------
CREATE TABLE `HospitalCode` (
  `id_hospital_code` int NOT NULL AUTO_INCREMENT,
  `id_hospital` int NOT NULL,
  `codigo` varchar(32) NOT NULL,
  `creado_por_id_usuario` int DEFAULT NULL,
  `usado_por_id_medico` int DEFAULT NULL,
  `expires_at` datetime DEFAULT NULL,
  `used_at` datetime DEFAULT NULL,
  `revoked_at` datetime DEFAULT NULL,
  `created_at` datetime NOT NULL DEFAULT current_timestamp(),
  `updated_at` datetime NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  PRIMARY KEY (`id_hospital_code`),
  UNIQUE KEY `uq_hospitalcode_codigo` (`codigo`),
  KEY `idx_hospitalcode_id_hospital` (`id_hospital`),
  KEY `idx_hospitalcode_creado_por` (`creado_por_id_usuario`),
  KEY `idx_hospitalcode_usado_por` (`usado_por_id_medico`),
  CONSTRAINT `fk_hospitalcode_hospital` FOREIGN KEY (`id_hospital`) REFERENCES `Hospital` (`id_hospital`),
  CONSTRAINT `fk_hospitalcode_usuario` FOREIGN KEY (`creado_por_id_usuario`) REFERENCES `Usuario` (`id_usuario`),
  CONSTRAINT `fk_hospitalcode_medico` FOREIGN KEY (`usado_por_id_medico`) REFERENCES `Medico` (`id_medico`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- -------------------------
-- ClinicalAudit
-- -------------------------
CREATE TABLE `ClinicalAudit` (
  `id_audit` int NOT NULL AUTO_INCREMENT,
  `entity_type` varchar(32) NOT NULL,
  `entity_id` int NOT NULL,
  `action` varchar(20) NOT NULL,
  `actor_id_usuario` int DEFAULT NULL,
  `before_json` text DEFAULT NULL,
  `after_json` text DEFAULT NULL,
  `meta_json` text DEFAULT NULL,
  `created_at` datetime NOT NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`id_audit`),
  KEY `idx_clinicalaudit_actor` (`actor_id_usuario`),
  KEY `idx_clinicalaudit_entity` (`entity_type`, `entity_id`),
  CONSTRAINT `fk_clinicalaudit_actor` FOREIGN KEY (`actor_id_usuario`) REFERENCES `Usuario` (`id_usuario`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- -------------------------
-- mensaje_gestion
-- -------------------------
CREATE TABLE `mensaje_gestion` (
  `id_mensaje` int NOT NULL,
  `asignado_admin_id_usuario` int DEFAULT NULL,
  `creado_en` datetime NOT NULL DEFAULT current_timestamp(),
  `actualizado_en` datetime NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  PRIMARY KEY (`id_mensaje`),
  KEY `idx_mensajegestion_admin` (`asignado_admin_id_usuario`),
  CONSTRAINT `fk_mensajegestion_mensaje` FOREIGN KEY (`id_mensaje`) REFERENCES `mensaje` (`id_mensaje`),
  CONSTRAINT `fk_mensajegestion_admin` FOREIGN KEY (`asignado_admin_id_usuario`) REFERENCES `Usuario` (`id_usuario`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- -------------------------
-- mensaje_evento
-- -------------------------
CREATE TABLE `mensaje_evento` (
  `id_evento` int NOT NULL AUTO_INCREMENT,
  `id_mensaje` int NOT NULL,
  `id_actor_usuario` int DEFAULT NULL,
  `accion` varchar(40) NOT NULL,
  `detalle_json` text DEFAULT NULL,
  `creado_en` datetime NOT NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`id_evento`),
  KEY `idx_mensajeevento_id_mensaje` (`id_mensaje`),
  KEY `idx_mensajeevento_actor` (`id_actor_usuario`),
  CONSTRAINT `fk_mensajeevento_mensaje` FOREIGN KEY (`id_mensaje`) REFERENCES `mensaje` (`id_mensaje`),
  CONSTRAINT `fk_mensajeevento_actor` FOREIGN KEY (`id_actor_usuario`) REFERENCES `Usuario` (`id_usuario`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- -------------------------
-- auth_login_attempt
-- -------------------------
CREATE TABLE `auth_login_attempt` (
  `id_attempt` int NOT NULL AUTO_INCREMENT,
  `correo` varchar(100) NOT NULL,
  `ip_address` varchar(64) DEFAULT NULL,
  `user_agent` varchar(300) DEFAULT NULL,
  `success` tinyint(1) NOT NULL DEFAULT 0,
  `reason` varchar(120) DEFAULT NULL,
  `created_at` datetime NOT NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`id_attempt`),
  KEY `idx_authlogin_correo_created` (`correo`, `created_at`),
  KEY `idx_authlogin_ip_created` (`ip_address`, `created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- -------------------------
-- payment_webhook_event
-- -------------------------
CREATE TABLE `payment_webhook_event` (
  `id_event` int NOT NULL AUTO_INCREMENT,
  `ref_payco` varchar(120) NOT NULL,
  `transaction_id` varchar(120) DEFAULT NULL,
  `estado` varchar(40) DEFAULT NULL,
  `firma_valida` tinyint(1) NOT NULL DEFAULT 0,
  `payload_json` text DEFAULT NULL,
  `processed` tinyint(1) NOT NULL DEFAULT 0,
  `attempts` int NOT NULL DEFAULT 0,
  `last_error` varchar(255) DEFAULT NULL,
  `created_at` datetime NOT NULL DEFAULT current_timestamp(),
  `updated_at` datetime NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  PRIMARY KEY (`id_event`),
  UNIQUE KEY `uq_webhook_ref_payco` (`ref_payco`),
  KEY `idx_webhook_processed_estado` (`processed`, `estado`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- -------------------------
-- clinical_note
-- -------------------------
CREATE TABLE `clinical_note` (
  `id_note` int NOT NULL AUTO_INCREMENT,
  `id_paciente` int NOT NULL,
  `id_medico` int NOT NULL,
  `segmento` varchar(60) NOT NULL DEFAULT 'GENERAL',
  `texto` text NOT NULL,
  `anchor_json` text DEFAULT NULL,
  `created_at` datetime NOT NULL DEFAULT current_timestamp(),
  `updated_at` datetime NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  PRIMARY KEY (`id_note`),
  KEY `idx_clinicalnote_paciente` (`id_paciente`),
  KEY `idx_clinicalnote_medico` (`id_medico`),
  KEY `idx_clinicalnote_segmento` (`segmento`),
  CONSTRAINT `fk_clinicalnote_paciente` FOREIGN KEY (`id_paciente`) REFERENCES `Paciente` (`id_paciente`),
  CONSTRAINT `fk_clinicalnote_medico` FOREIGN KEY (`id_medico`) REFERENCES `Medico` (`id_medico`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- =========================
-- PROCEDIMIENTOS + EVENTOS + TRIGGERS DE REGLAS DE NEGOCIO
-- =========================
DROP EVENT IF EXISTS ev_reconcile_subscriptions_daily_2300;
DROP PROCEDURE IF EXISTS sp_activate_subscription;
DROP PROCEDURE IF EXISTS sp_reconcile_expired_subscriptions;
DROP TRIGGER IF EXISTS trg_pago_bi;
DROP TRIGGER IF EXISTS trg_susc_bi;
DROP TRIGGER IF EXISTS trg_susc_bu;

DELIMITER $$

CREATE PROCEDURE sp_activate_subscription(IN p_suscripcion_id INT, IN p_now DATETIME)
BEGIN
  DECLARE v_id_medico INT;
  DECLARE v_id_hospital INT;
  DECLARE v_id_plan INT;
  DECLARE v_meses INT DEFAULT 1;
  DECLARE v_exp DATETIME;
  DECLARE v_base DATETIME;

  SELECT s.id_medico, s.id_hospital, s.id_plan, s.fecha_expiracion
    INTO v_id_medico, v_id_hospital, v_id_plan, v_exp
  FROM Suscripcion s
  WHERE s.id_suscripcion = p_suscripcion_id
  FOR UPDATE;

  IF v_id_plan IS NULL THEN
    SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'Suscripcion no existe o no tiene plan.';
  END IF;

  IF (v_id_medico IS NULL AND v_id_hospital IS NULL) OR (v_id_medico IS NOT NULL AND v_id_hospital IS NOT NULL) THEN
    SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'Suscripcion invalida: debe existir exactamente un pagador.';
  END IF;

  SELECT p.duracion_meses INTO v_meses
  FROM Plan p
  WHERE p.id_plan = v_id_plan;

  IF v_meses IS NULL OR v_meses < 1 THEN
    SET v_meses = 1;
  END IF;

  SET v_base = GREATEST(IFNULL(v_exp, p_now), p_now);

  UPDATE Suscripcion
  SET
    estado = 'ACTIVA',
    fecha_inicio = p_now,
    fecha_expiracion = DATE_ADD(v_base, INTERVAL v_meses MONTH)
  WHERE id_suscripcion = p_suscripcion_id;

  IF v_id_medico IS NOT NULL THEN
    UPDATE Medico m
      JOIN Usuario u ON u.id_usuario = m.id_usuario
    SET
      m.estado = 'ACTIVO',
      u.activo = 1
    WHERE m.id_medico = v_id_medico;
  END IF;

  IF v_id_hospital IS NOT NULL THEN
    UPDATE Hospital h
    SET h.estado = 'ACTIVO'
    WHERE h.id_hospital = v_id_hospital;

    UPDATE Medico m
      JOIN Usuario u ON u.id_usuario = m.id_usuario
    SET
      m.estado = 'ACTIVO',
      u.activo = 1
    WHERE m.id_hospital = v_id_hospital;
  END IF;
END$$

CREATE PROCEDURE sp_reconcile_expired_subscriptions(IN p_now DATETIME)
BEGIN
  DROP TEMPORARY TABLE IF EXISTS tmp_expired_subscriptions;
  CREATE TEMPORARY TABLE tmp_expired_subscriptions (
    id_suscripcion INT PRIMARY KEY,
    id_medico INT NULL,
    id_hospital INT NULL
  ) ENGINE=Memory;

  INSERT INTO tmp_expired_subscriptions (id_suscripcion, id_medico, id_hospital)
  SELECT s.id_suscripcion, s.id_medico, s.id_hospital
  FROM Suscripcion s
  WHERE
    s.estado = 'ACTIVA'
    AND s.fecha_expiracion IS NOT NULL
    AND s.fecha_expiracion <= p_now;

  UPDATE Suscripcion s
    JOIN tmp_expired_subscriptions t ON t.id_suscripcion = s.id_suscripcion
  SET s.estado = 'PAUSADA';

  UPDATE Hospital h
    JOIN (SELECT DISTINCT id_hospital FROM tmp_expired_subscriptions WHERE id_hospital IS NOT NULL) th
      ON th.id_hospital = h.id_hospital
  SET h.estado = 'INACTIVO';

  UPDATE Medico m
    JOIN Usuario u ON u.id_usuario = m.id_usuario
    JOIN (SELECT DISTINCT id_hospital FROM tmp_expired_subscriptions WHERE id_hospital IS NOT NULL) th
      ON th.id_hospital = m.id_hospital
  SET
    m.estado = 'INACTIVO',
    u.activo = 0;

  UPDATE Medico m
    JOIN Usuario u ON u.id_usuario = m.id_usuario
    JOIN (
      SELECT DISTINCT id_medico
      FROM tmp_expired_subscriptions
      WHERE id_medico IS NOT NULL AND id_hospital IS NULL
    ) tm ON tm.id_medico = m.id_medico
  SET
    m.estado = 'INACTIVO',
    u.activo = 0
  WHERE NOT EXISTS (
      SELECT 1
      FROM Suscripcion sa
      WHERE
        sa.id_medico = m.id_medico
        AND sa.estado = 'ACTIVA'
    )
    AND NOT EXISTS (
      SELECT 1
      FROM Suscripcion sh
      WHERE
        sh.id_hospital = m.id_hospital
        AND sh.estado = 'ACTIVA'
    );
END$$

CREATE EVENT ev_reconcile_subscriptions_daily_2300
ON SCHEDULE EVERY 1 DAY
STARTS (TIMESTAMP(CURRENT_DATE, '23:00:00') + INTERVAL 1 DAY)
DO
BEGIN
  -- Requiere event_scheduler=ON en el servidor MySQL.
  CALL sp_reconcile_expired_subscriptions(UTC_TIMESTAMP());
END$$

CREATE TRIGGER trg_pago_bi
BEFORE INSERT ON Pago
FOR EACH ROW
BEGIN
  DECLARE v_precio DECIMAL(12,2);
  DECLARE v_plan   INT;
  DECLARE v_msg    VARCHAR(255);

  SELECT id_plan INTO v_plan
  FROM Suscripcion
  WHERE id_suscripcion = NEW.id_suscripcion;

  IF v_plan IS NULL THEN
    SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'Suscripcion no existe.';
  END IF;

  SELECT precio INTO v_precio
  FROM Plan
  WHERE id_plan = v_plan;

  IF v_precio IS NULL THEN
    SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'Plan asociado a la suscripcion no encontrado.';
  END IF;

  IF NEW.monto < v_precio THEN
    SET v_msg = CONCAT('Monto insuficiente. Precio del plan: ', v_precio);
    SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = v_msg;
  END IF;
END$$

CREATE TRIGGER trg_susc_bi
BEFORE INSERT ON Suscripcion
FOR EACH ROW
BEGIN
  IF (NEW.id_medico IS NULL AND NEW.id_hospital IS NULL) OR
     (NEW.id_medico IS NOT NULL AND NEW.id_hospital IS NOT NULL) THEN
    SIGNAL SQLSTATE '45000'
      SET MESSAGE_TEXT = 'Debe asignarse exactamente un pagador: id_medico o id_hospital.';
  END IF;
END$$

CREATE TRIGGER trg_susc_bu
BEFORE UPDATE ON Suscripcion
FOR EACH ROW
BEGIN
  IF (NEW.id_medico IS NULL AND NEW.id_hospital IS NULL) OR
     (NEW.id_medico IS NOT NULL AND NEW.id_hospital IS NOT NULL) THEN
    SIGNAL SQLSTATE '45000'
      SET MESSAGE_TEXT = 'Debe asignarse exactamente un pagador: id_medico o id_hospital.';
  END IF;

  IF NEW.fecha_expiracion IS NOT NULL AND NEW.fecha_inicio IS NOT NULL THEN
    IF NEW.fecha_expiracion <= NEW.fecha_inicio THEN
      SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'fecha_expiracion debe ser posterior a fecha_inicio.';
    END IF;
  END IF;
END$$

DELIMITER ;

-- =========================
-- SEMILLA MINIMA (SOLO MEDICO + SUSCRIPCION + PACIENTES)
-- =========================
INSERT INTO `Usuario` (
  `nombre`, `apellido`, `correo`, `contrasena`, `rol`, `activo`
) VALUES (
  'Admin',
  'Root',
  'admin@root.com',
  '$2b$12$v8VyPk78PQs/hOVEG.BPAOwRbU19yUuXiiidy9eYYdO2QY0cakZpK',
  'ADMINISTRADOR',
  1
);

-- Forzar password del administrador semilla: Administrador.1
UPDATE `Usuario`
SET `contrasena` = '$2b$12$hmI2sKWc9PAroTqofiUhl.kiHG84ZSkw57x29t9rlw4EQyu9MoOoO'
WHERE `correo` = 'admin@root.com';

INSERT INTO `Plan` (`nombre`, `precio`, `periodo`, `duracion_meses`) VALUES
('Basico Mensual',          39900.00,  'MENSUAL',    1),
('Profesional Mensual',     69900.00,  'MENSUAL',    1),
('Profesional Trimestral', 179900.00,  'TRIMESTRAL', 3),
('Empresarial Anual',      599900.00,  'ANUAL',     12);

-- Medico de pruebas (sin hospital)
INSERT INTO `Usuario` (
  `nombre`, `apellido`, `correo`, `contrasena`, `telefono`, `ciudad`, `rol`, `activo`
) VALUES (
  'Medico',
  'Pruebas',
  'medico.pruebas@som3d.local',
  '$2b$12$HSzmmzrqaGgBVH3aGMmcEONKEULLPAqTgzHl8IWmfOqAtN4G9jS5m',
  '3000000000',
  'Bogota',
  'MEDICO',
  1
);

INSERT INTO `Medico` (`id_usuario`, `id_hospital`, `referenciado`, `estado`)
SELECT `id_usuario`, NULL, 0, 'ACTIVO'
FROM `Usuario`
WHERE `correo` = 'medico.pruebas@som3d.local'
LIMIT 1;

-- Muchos pacientes de prueba (120) para el medico anterior.
-- No se crean estudios, jobs ni STL.
INSERT INTO `Paciente` (
  `id_medico`,
  `doc_tipo`,
  `doc_numero`,
  `nombres`,
  `apellidos`,
  `fecha_nacimiento`,
  `sexo`,
  `telefono`,
  `correo`,
  `direccion`,
  `ciudad`,
  `estado`
)
WITH RECURSIVE seq AS (
  SELECT 1 AS n
  UNION ALL
  SELECT n + 1 FROM seq WHERE n < 120
)
SELECT
  m.`id_medico`,
  'CC',
  CONCAT('PRUEBA-', LPAD(seq.n, 5, '0')),
  CONCAT('Paciente ', LPAD(seq.n, 3, '0')),
  'Carga',
  DATE_SUB(CURDATE(), INTERVAL (18 + (seq.n % 50)) YEAR),
  IF(seq.n % 2 = 0, 'M', 'F'),
  CONCAT('300', LPAD(seq.n, 7, '0')),
  CONCAT('paciente', LPAD(seq.n, 3, '0'), '@example.test'),
  CONCAT('Direccion ', seq.n),
  'Bogota',
  'ACTIVO'
FROM seq
CROSS JOIN (
  SELECT md.`id_medico`
  FROM `Medico` md
  INNER JOIN `Usuario` u ON u.`id_usuario` = md.`id_usuario`
  WHERE u.`correo` = 'medico.pruebas@som3d.local'
  LIMIT 1
) AS m;

-- Suscripcion ACTIVA para el medico de pruebas
INSERT INTO `Suscripcion` (
  `id_medico`,
  `id_hospital`,
  `id_plan`,
  `fecha_inicio`,
  `estado`
)
SELECT
  m.`id_medico`,
  NULL,
  p.`id_plan`,
  NOW(),
  'ACTIVA'
FROM `Medico` m
INNER JOIN `Usuario` u ON u.`id_usuario` = m.`id_usuario`
INNER JOIN `Plan` p ON p.`nombre` = 'Profesional Mensual'
WHERE u.`correo` = 'medico.pruebas@som3d.local'
LIMIT 1;

-- Pago inicial de semilla
INSERT INTO `Pago` (
  `id_suscripcion`,
  `referencia_epayco`,
  `monto`,
  `fecha_pago`
)
SELECT
  s.`id_suscripcion`,
  CONCAT('SEED-PAGO-', s.`id_suscripcion`, '-', DATE_FORMAT(NOW(), '%Y%m%d%H%i%s')),
  p.`precio`,
  NOW()
FROM `Suscripcion` s
INNER JOIN `Plan` p ON p.`id_plan` = s.`id_plan`
INNER JOIN `Medico` m ON m.`id_medico` = s.`id_medico`
INNER JOIN `Usuario` u ON u.`id_usuario` = m.`id_usuario`
WHERE u.`correo` = 'medico.pruebas@som3d.local'
  AND s.`estado` = 'ACTIVA'
ORDER BY s.`id_suscripcion` DESC
LIMIT 1;

-- Activar/renovar suscripcion con logica oficial de negocio (SP)
CALL sp_activate_subscription(
  (
    SELECT s.`id_suscripcion`
    FROM `Suscripcion` s
    INNER JOIN `Medico` m ON m.`id_medico` = s.`id_medico`
    INNER JOIN `Usuario` u ON u.`id_usuario` = m.`id_usuario`
    WHERE u.`correo` = 'medico.pruebas@som3d.local'
    ORDER BY s.`id_suscripcion` DESC
    LIMIT 1
  ),
  NOW()
);


