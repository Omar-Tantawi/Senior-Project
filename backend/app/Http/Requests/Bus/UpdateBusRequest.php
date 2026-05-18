<?php

namespace App\Http\Requests\Bus;

use Illuminate\Foundation\Http\FormRequest;

class UpdateBusRequest extends FormRequest
{
    public function authorize(): bool { return true; }

    public function rules(): array
    {
        return [
            'plate_number' => 'sometimes|string|max:50|unique:bus,plate_number,' . $this->route('id') . ',bus_id',
        ];
    }
}
